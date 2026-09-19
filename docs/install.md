# Установка Linux-машины

Одна команда ставит всё, на чём платформа работает с симуляторами, обучением и роботами:
[`scripts/install.sh`](../scripts/install.sh) ([ADR-0071](decisions/0071-one-installer-packages-install-themselves.md)).
Что лежит в `scripts/` и зачем — [scripts/README.md](../scripts/README.md).

Профили, валидатор, генераторы и весь CLI работают и без этого — см. [README](../README.md).
Установка нужна для запуска симуляции, обучения и прогонов на железе.

Требования: Ubuntu 24.04, x86_64; для обучения — GPU NVIDIA. Версия ОС жёсткая: ROS 2 Jazzy
собирается только под noble, это зафиксировано в [ADR-0005](decisions/0005-training-resources.md).

## Быстрый путь

```bash
git clone https://github.com/FastR1se/heron.git ~/heron && cd ~/heron
scripts/install.sh install                            # платформа и пакеты роботов
scripts/install.sh install train --accept-nvidia-eula # стек обучения, десятки ГиБ
scripts/install.sh verify                             # всё запускается по-настоящему
scripts/install.sh status                             # что где стоит
```

Дальше каждый терминал начинается с окружения под задачу:

```bash
source scripts/activate.sh exec         # validate, gen, mission run; мир MuJoCo
source scripts/activate.sh ros          # мир Gazebo, робот на ros2_control
source scripts/activate.sh train        # обучение
source scripts/activate.sh unitree_g1   # процесс G1 (без ROS 2 в оболочке)
source scripts/activate.sh robopro_rc   # процесс RC5 на стенде
```

## Команды

| Команда | Что делает |
|---|---|
| `install [компонент...]` | ставит; без аргументов — всё, кроме `train` и `docker` |
| `check [компонент...]` | предусловия этих компонентов, ничего не меняя |
| `verify [компонент...]` | запускает поставленное: шаги Gazebo и MuJoCo, импорты адаптера, пустую сцену Isaac, сборки пакетов |
| `sync [окружение...]` | заново ставит платформу и пакеты в окружения (после `git pull`); без sudo и сборок |
| `status` | таблица: компонент, есть ли, где и какой версии |
| `report` | версии всех сред таблицей для карточки эксперимента |

Флаги: `--dry-run` печатает команды, ничего не выполняя; `--accept-nvidia-eula` — согласие с лицензией
NVIDIA, без него `train` не ставится.

## Компоненты

| Компонент | Что ставит | Где |
|---|---|---|
| `base` | пакеты сборки, `vulkan-tools`, `ffmpeg`, `uv` | система |
| `ros2` | ROS 2 Jazzy desktop, `ros2_control`, контроллеры, `xacro`, MCAP-хранилище rosbag2 | `/opt/ros/jazzy` |
| `gazebo` | Gazebo Harmonic, мосты `ros_gz` и `gz_ros2_control` | система |
| `exec` | платформа с группами `sim dev ros2 record onnx` и все пакеты `examples/plugins` | `.venv` репозитория |
| `descriptions` | описания производителей для примеров (`examples/robots/*/fetch_description.py`) | `examples/robots/*/vendor` |
| `train` | Isaac Sim 5.1, Isaac Lab 2.3, rsl_rl, платформа и пакеты | `~/train-stack` (`TRAIN_ROOT`) |
| `docker` | Docker CE, compose, NVIDIA Container Toolkit | система |
| `unitree_g1` | CycloneDDS 0.10.2, venv адаптера с SDK, мост Gazebo, двойник MuJoCo | `~/robot-stack/unitree_g1` (`STACK_ROOT`) |
| `robopro_rc` | venv стенда с API производителя (нужен `API_DIR` с архивом) | `~/robot-stack/robopro_rc` |

Последние две строки — не компоненты платформы: это установщики пакетов роботов,
`examples/plugins/<пакет>/install.sh`. Платформа их находит и запускает, но не знает, что внутри:
окружение пакета объявляет и строит сам пакет ([ADR-0022](decisions/0022-vendor-dds-environment.md),
[ADR-0034](decisions/0034-gazebo-scene-dds-environment-and-names.md)). Свой пакет получает установку
так же: положите рядом с его `pyproject.toml` `install.sh` с командами `install`, `sync`, `verify`,
`status` — образец [examples/plugins/robopro_rc/install.sh](../examples/plugins/robopro_rc/install.sh).

Каждый компонент идемпотентен: повторный запуск ничего не переставляет. Проверка места считает только
то, что предстоит поставить.

## Окружения Python — их несколько, и так задумано

| Окружение | Кто в нём работает | Почему отдельно |
|---|---|---|
| `.venv` | CLI, мир, координатор, робот на `ros2_control` | Python той же версии, что у ROS 2: `rclpy` — скомпилированное расширение |
| `~/train-stack/venv` | обучение | Isaac требует Python 3.11; ROS 2 в обучение не попадает (инвариант 2) |
| `~/robot-stack/unitree_g1/venv` | процесс G1 | CycloneDDS 0.10.2 и ни одной `libddsc` от ROS 2 ([ADR-0022](decisions/0022-vendor-dds-environment.md)) |
| `~/robot-stack/robopro_rc/venv` | процесс RC5 | API производителя со своими зависимостями |

**В `.venv` не выполняйте `uv sync`**: он удаляет всё, чего нет в `pyproject.toml`, — пакеты роботов
вместе с этим. Обновить окружения после `git pull` — `scripts/install.sh sync`.

## Что делает человек, а не скрипт

Скрипт ничего не чинит молча: нарушенное предусловие останавливает работу и печатает команду, которую
выполняют руками.

**Драйвер NVIDIA** — `sudo ubuntu-drivers install` и перезагрузка. Ставится отдельно: его версия
входит в условия каждого эксперимента (раздел 13 архитектуры).

**Подкачка не меньше 16 ГиБ** при 15 ГиБ RAM — только для `train`
([ADR-0005](decisions/0005-training-resources.md)). Достаточно файла на корне:

```bash
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Архив API «Робопро»** — не на индексе и без лицензии на распространение: распакуйте и укажите
`API_DIR=<каталог> scripts/install.sh install robopro_rc`. Без архива пакет пропускается, остальное
ставится.

**Замена закреплённого коммита.** Если в каталоге исходника другой коммит, установщик останавливается
и не перезаписывает его: закреплённые версии — решения (ADR-0022, ADR-0023). Замена — удалить каталог
и поставить заново.

## Особенности, на которые наступали

- `OMNI_KIT_ACCEPT_EULA=YES` — переменная окружения, а не аргумент: ставится **перед** командой.
  Для проверки её выставляет сам скрипт.
- Первый запуск Isaac Sim компилирует шейдеры — десятки минут тишины, похожей на зависание.
- `isaaclab.sh` ставит модули по очереди и **завершается успешно, даже когда один из них не
  собрался**. Скрипт проверяет набор пакетов после установки.
- `flatdict` ставится без изоляции сборки: его `setup.py` импортирует `pkg_resources`, которого нет в
  setuptools 81 и новее. Без этого не собирается и сам `isaaclab`.
- Isaac Lab поднимает `psutil` и `typing_extensions` выше пинов `isaacsim-kernel`. Скрипт печатает
  конфликты предупреждением: существенны ли они, показывает `verify train`.
- Двойник MuJoCo производителя слинкован со своей `libddsc`, а оболочка с ROS 2 держит в
  `LD_LIBRARY_PATH` библиотеку Jazzy. Поэтому на `PATH` лежит запускатель
  `~/robot-stack/unitree_g1/bin/unitree_mujoco`: он ставит библиотеки двойника первыми только для
  своего процесса и заменяет себя двоичным файлом (сигналы мира доходят, ADR-0037).

## Переход со старых каталогов

До ADR-0071 окружения жили в каталогах этапов: `~/phase4-probe` (venv адаптера G1, CycloneDDS),
`~/phase4-stage1` (двойник), `~/phase4-stage5/ws` (мост), `~/g1` (ручная установка для железа).
Установщик их не читает и не трогает. После `scripts/install.sh install unitree_g1` и
`scripts/install.sh verify unitree_g1` их можно удалить руками; записи прогонов в `~/phase*/` —
данные экспериментов, их стоит сохранить. `~/train-stack` остаётся на месте и используется как есть.

## Состояние

Установщик переписан на Windows 19.09.2026: проверены синтаксис (`bash -n`), справка, сухой прогон
разбора компонентов. Команды внутри компонентов — те же, что прошли на Linux в фазах 0–8 (скрипты
этапов, журналы 13–19.09.2026), но собраны в новом порядке и в новых каталогах: первый прогон на
Linux — проверка установщика.
