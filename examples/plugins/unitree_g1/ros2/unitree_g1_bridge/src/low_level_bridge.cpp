// The Gazebo twin of a G1's drive firmware (ADR-0033).
//
// On the robot, every motor evaluates the law of the last LowCmd it received:
//   tau = tau_ff + kp * (q* - q) + kd * (dq* - dq).
// Here the controller manager does, in update(), in step with the physics: LowCmd in,
// effort out, LowState back. The vendor's MuJoCo twin is built the same way -- MuJoCo
// plus a bridge -- and this controller follows it, with one difference: a slot whose
// mode is 0 gets no torque, as a disabled drive of the robot does.
//
// Nothing here knows which joint sits in which slot, which body this is or what the
// topics are called. The package's Python side writes all of it into the parameters
// below from the same tables the DDS adapter reads (ADR-0027), and a test of the
// package holds these names against what it writes.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

namespace unitree_g1_bridge
{

using controller_interface::CallbackReturn;
using controller_interface::interface_configuration_type;
using controller_interface::InterfaceConfiguration;
using controller_interface::return_type;
using unitree_hg::msg::LowCmd;
using unitree_hg::msg::LowState;

// Parameters, as unitree_g1.bridge writes them into the controller manager's file.
constexpr char kJointsParam[] = "joints";
constexpr char kSlotsParam[] = "slots";
constexpr char kMachineParam[] = "mode_machine";
constexpr char kCommandTopicParam[] = "command_topic";
constexpr char kStateTopicParam[] = "state_topic";
constexpr char kImuSensorParam[] = "imu_sensor";

// The IMU's state interfaces, in the order LowState carries them: quaternion w, x, y, z,
// then gyroscope x, y, z (ADR-0076). unitree_g1.bridge.IMU_INTERFACES, held by a test.
constexpr std::array<const char *, 7> kImuInterfaces = {
  "orientation.w", "orientation.x", "orientation.y", "orientation.z",
  "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"};

constexpr std::size_t kSlotCount = std::tuple_size<LowCmd::_motor_cmd_type>::value;
static_assert(
  kSlotCount == std::tuple_size<LowState::_motor_state_type>::value,
  "LowCmd and LowState must address the same motor slots");

// "1: Enable, 0: Disable" -- the vendor's g1_low_level_example.py.
constexpr std::uint8_t kMotorDisabled = 0;

// Warnings about a command that cannot be evaluated, at most once per this many ms.
constexpr int kWarnPeriodMs = 1000;

template<typename Interfaces>
std::optional<std::size_t> index_of(
  const Interfaces & interfaces, const std::string & joint, const std::string & interface)
{
  for (std::size_t index = 0; index < interfaces.size(); ++index) {
    if (
      interfaces[index].get_prefix_name() == joint &&
      interfaces[index].get_interface_name() == interface)
    {
      return index;
    }
  }
  return std::nullopt;
}

class LowLevelBridge : public controller_interface::ControllerInterface
{
public:
  CallbackReturn on_init() override
  {
    auto_declare<std::vector<std::string>>(kJointsParam, std::vector<std::string>{});
    auto_declare<std::vector<std::int64_t>>(kSlotsParam, std::vector<std::int64_t>{});
    auto_declare<std::int64_t>(kMachineParam, -1);
    auto_declare<std::string>(kCommandTopicParam, std::string{});
    auto_declare<std::string>(kStateTopicParam, std::string{});
    auto_declare<std::string>(kImuSensorParam, std::string{});
    return CallbackReturn::SUCCESS;
  }

  // Everything is checked here, before a single interface is claimed: a parameter
  // file that does not describe a link refuses the controller with the reason,
  // instead of driving the wrong motors.
  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    const auto node = get_node();
    const auto refuse = [&node](const std::string & reason) {
        RCLCPP_ERROR(node->get_logger(), "%s", reason.c_str());
        return CallbackReturn::ERROR;
      };

    joints_ = node->get_parameter(kJointsParam).as_string_array();
    const auto slots = node->get_parameter(kSlotsParam).as_integer_array();
    const auto machine = node->get_parameter(kMachineParam).as_int();
    const auto command_topic = node->get_parameter(kCommandTopicParam).as_string();
    const auto state_topic = node->get_parameter(kStateTopicParam).as_string();
    imu_sensor_ = node->get_parameter(kImuSensorParam).as_string();

    if (joints_.empty()) {
      return refuse("parameter 'joints' is empty: there is nothing to drive");
    }
    if (slots.size() != joints_.size()) {
      return refuse(
        "parameter 'slots' has " + std::to_string(slots.size()) + " values for " +
        std::to_string(joints_.size()) + " joints");
    }
    std::array<bool, kSlotCount> taken{};
    slots_.clear();
    for (std::size_t index = 0; index < slots.size(); ++index) {
      const auto slot = slots[index];
      if (slot < 0 || static_cast<std::size_t>(slot) >= kSlotCount) {
        return refuse(
          "joint " + joints_[index] + ": slot " + std::to_string(slot) + " outside 0.." +
          std::to_string(kSlotCount - 1));
      }
      if (taken[static_cast<std::size_t>(slot)]) {
        return refuse(
          "joint " + joints_[index] + ": slot " + std::to_string(slot) +
          " already belongs to another joint");
      }
      taken[static_cast<std::size_t>(slot)] = true;
      slots_.push_back(static_cast<std::size_t>(slot));
    }
    if (machine < 0 || machine > 255) {
      return refuse("parameter 'mode_machine' must be 0..255, got " + std::to_string(machine));
    }
    if (command_topic.empty() || state_topic.empty()) {
      return refuse("parameters 'command_topic' and 'state_topic' must name topics");
    }
    if (imu_sensor_.empty()) {
      return refuse("parameter 'imu_sensor' must name the IMU: LowState always carries imu_state");
    }

    subscription_ = node->create_subscription<LowCmd>(
      command_topic, rclcpp::SystemDefaultsQoS(),
      [this](const LowCmd::SharedPtr message) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = *message;
        fresh_ = true;
      });
    publisher_ = node->create_publisher<LowState>(state_topic, rclcpp::SystemDefaultsQoS());

    state_ = LowState{};
    state_.mode_machine = static_cast<std::uint8_t>(machine);

    RCLCPP_INFO(
      node->get_logger(), "%zu joints: %s -> effort, state -> %s, mode_machine %ld",
      joints_.size(), command_topic.c_str(), state_topic.c_str(), static_cast<long>(machine));
    return CallbackReturn::SUCCESS;
  }

  InterfaceConfiguration command_interface_configuration() const override
  {
    InterfaceConfiguration configuration{interface_configuration_type::INDIVIDUAL, {}};
    for (const auto & joint : joints_) {
      configuration.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
    }
    return configuration;
  }

  InterfaceConfiguration state_interface_configuration() const override
  {
    InterfaceConfiguration configuration{interface_configuration_type::INDIVIDUAL, {}};
    for (const auto & joint : joints_) {
      configuration.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
      configuration.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
    }
    for (const auto * name : kImuInterfaces) {
      configuration.names.push_back(imu_sensor_ + "/" + name);
    }
    return configuration;
  }

  // Interfaces are found by name, not by the order they were loaned in.
  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    command_index_.clear();
    position_index_.clear();
    velocity_index_.clear();
    for (const auto & joint : joints_) {
      const auto command = index_of(command_interfaces_, joint, hardware_interface::HW_IF_EFFORT);
      const auto position =
        index_of(state_interfaces_, joint, hardware_interface::HW_IF_POSITION);
      const auto velocity =
        index_of(state_interfaces_, joint, hardware_interface::HW_IF_VELOCITY);
      if (!command || !position || !velocity) {
        RCLCPP_ERROR(
          get_node()->get_logger(),
          "joint %s: effort command, position or velocity state interface was not loaned",
          joint.c_str());
        return CallbackReturn::ERROR;
      }
      command_index_.push_back(*command);
      position_index_.push_back(*position);
      velocity_index_.push_back(*velocity);
    }
    for (std::size_t index = 0; index < kImuInterfaces.size(); ++index) {
      const auto found = index_of(state_interfaces_, imu_sensor_, kImuInterfaces[index]);
      if (!found) {
        RCLCPP_ERROR(
          get_node()->get_logger(), "IMU %s: state interface %s was not loaned",
          imu_sensor_.c_str(), kImuInterfaces[index]);
        return CallbackReturn::ERROR;
      }
      imu_index_[index] = *found;
    }

    // A command that arrived before activation is not one this controller was asked
    // to evaluate.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fresh_ = false;
    }
    has_command_ = false;
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    for (auto & interface : command_interfaces_) {
      (void)interface.set_value(0.0);
    }
    return CallbackReturn::SUCCESS;
  }

  return_type update(const rclcpp::Time & time, const rclcpp::Duration &) override
  {
    // The subscription's thread holds the lock only to copy one message. If it holds
    // it now, the previous command stands for one more step.
    {
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (lock.owns_lock() && fresh_) {
        command_ = pending_;
        fresh_ = false;
        has_command_ = true;
      }
    }

    for (std::size_t index = 0; index < joints_.size(); ++index) {
      const std::size_t slot = slots_[index];
      const double q = state_interfaces_[position_index_[index]].get_value();
      const double dq = state_interfaces_[velocity_index_[index]].get_value();

      // No command yet: no torque, the robot falls -- as in the vendor's twin.
      double tau = 0.0;
      std::uint8_t mode = kMotorDisabled;
      if (has_command_) {
        const auto & motor = command_.motor_cmd[slot];
        mode = motor.mode;
        if (motor.mode != kMotorDisabled) {
          tau = motor.tau + motor.kp * (motor.q - q) + motor.kd * (motor.dq - dq);
        }
      }
      if (!std::isfinite(tau)) {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), kWarnPeriodMs,
          "joint %s (slot %zu): the command does not evaluate to a number; torque 0",
          joints_[index].c_str(), slot);
        tau = 0.0;
      }
      (void)command_interfaces_[command_index_[index]].set_value(tau);

      auto & reported = state_.motor_state[slot];
      reported.mode = mode;
      reported.q = static_cast<float>(q);
      reported.dq = static_cast<float>(dq);
      // What the bridge ordered; the vendor's twin reports the actuator's force.
      reported.tau_est = static_cast<float>(tau);
    }

    // The pelvis IMU, as the robot reports it: quaternion w, x, y, z; gyroscope x, y, z.
    for (std::size_t index = 0; index < 4; ++index) {
      state_.imu_state.quaternion[index] =
        static_cast<float>(state_interfaces_[imu_index_[index]].get_value());
    }
    for (std::size_t index = 0; index < 3; ++index) {
      state_.imu_state.gyroscope[index] =
        static_cast<float>(state_interfaces_[imu_index_[4 + index]].get_value());
    }

    // Milliseconds of world time, as the vendor's twin counts them; wraps like uint32.
    state_.tick = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(std::llround(time.seconds() * 1000.0)));
    publisher_->publish(state_);
    return return_type::OK;
  }

private:
  std::vector<std::string> joints_;
  std::vector<std::size_t> slots_;
  std::vector<std::size_t> command_index_;
  std::vector<std::size_t> position_index_;
  std::vector<std::size_t> velocity_index_;
  std::string imu_sensor_;
  std::array<std::size_t, kImuInterfaces.size()> imu_index_{};

  rclcpp::Subscription<LowCmd>::SharedPtr subscription_;
  rclcpp::Publisher<LowState>::SharedPtr publisher_;

  std::mutex mutex_;
  LowCmd pending_;      // guarded by mutex_
  bool fresh_ = false;  // guarded by mutex_

  LowCmd command_;  // update() only
  bool has_command_ = false;
  LowState state_;
};

}  // namespace unitree_g1_bridge

PLUGINLIB_EXPORT_CLASS(unitree_g1_bridge::LowLevelBridge, controller_interface::ControllerInterface)
