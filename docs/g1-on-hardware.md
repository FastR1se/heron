# Unitree G1: команды на настоящего робота

Пошаговая инструкция: от пустой машины до движения живого G1 через платформу. Каждый шаг — копипаст
и проверка, что он удался.

Обзор платформы — [руководство пользователя](user-guide.md); про G1 в симуляторах — раздел
[«Запустить G1 в MuJoCo и в Gazebo»](user-guide.md#19-запустить-g1-в-mujoco-и-в-gazebo) и
[README пакета](../examples/plugins/unitree_g1/README.md).

> **Что проверено, а что нет.** На настоящем G1 платформа ещё не запускалась ни разу. Проверены
> двойник MuJoCo производителя и мост в Gazebo (фаза 4, [эксперимент 0001](experiments/0001-g1-dds-twins.md)).
> Код связи с железом и с двойником **один и тот же** — отличаются только `domain` и `interface`
> ([ADR-0020](decisions/0020-humanoid-joins-the-system-through-vendor-dds.md),
> [ADR-0029](decisions/0029-g1-dds-channel.md)), — но первый живой запуск всё равно первый.
> Прогон на железе как цель (`--target real`) введён в фазе 5 и пока проверялся только на `dummy`.

Всё делается на **Linux-машине** (Ubuntu 24.04). На Windows — только проверки без робота
(шаг 10).

Обозначения: `~/heron` — клон репозитория, `~/g1` — новый каталог под всё вендорское.

---

## Содержание

- [Часть 1. Зависимости](#часть-1-зависимости-делается-один-раз)
- [Часть 2. Настройка под ваш экземпляр](#часть-2-настройка-под-ваш-экземпляр)
- [Часть 3. Робот](#часть-3-робот)
- [Частые ошибки](#частые-ошибки)
- [Что происходит внутри](#что-происходит-внутри)

---

# Часть 1. Зависимости (делается один раз)

## Почему это выглядит сложно

Три факта, из которых растёт вся возня:

1. Робот говорит по DDS библиотекой **CycloneDDS 0.10.2**. Готовых пакетов (колёс) этой версии для
   Python 3.12 нет — её надо собрать из исходников.
2. Версия **0.11 и новее запрещена** ([ADR-0022](decisions/0022-vendor-dds-environment.md)). Не
   «нежелательна»: её участник, обнаружив в сети робота на 0.10.2, роняет процессы робота. Поэтому
   `pip install cyclonedds` без версии — прямой путь к поломке.
3. Платформа **не опубликована в PyPI**. Пакет G1 объявляет зависимость `heron>=0.3`, и обычная
   установка пошла бы искать её в интернете и упала. Лечится флагом `--no-deps` и установкой
   настоящих зависимостей ядра руками.

Поэтому робот получает **отдельный venv**, никак не связанный с `.venv` репозитория.

## Шаг 1. Системные пакеты

```bash
sudo apt update
sudo apt install -y git cmake build-essential bison python3-dev python3.12-venv libssl-dev
```

## Шаг 2. Исходники на закреплённых версиях

```bash
mkdir -p ~/g1 && cd ~/g1
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git cyclonedds-src
git -C cyclonedds-src checkout 0.10.2

git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git -C unitree_sdk2_python checkout 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
```

Коммит SDK закреплён намеренно: на нём всё проверялось в фазе 4.

## Шаг 3. Собрать C-библиотеку CycloneDDS

```bash
cmake -S ~/g1/cyclonedds-src -B ~/g1/cyclonedds-build \
      -DCMAKE_INSTALL_PREFIX=$HOME/g1/cyclonedds-install \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SHM=OFF
cmake --build ~/g1/cyclonedds-build --target install -j$(nproc)
```

Проверка — файл обязан существовать:

```bash
ls ~/g1/cyclonedds-install/lib/libddsc.so || ls ~/g1/cyclonedds-install/lib64/libddsc.so
```

Нет файла — дальше идти бессмысленно, причина в выводе сборки.

## Шаг 4. Venv робота: обвязка Python и SDK

```bash
python3.12 -m venv ~/g1/venv
source ~/g1/venv/bin/activate
```

**В этом терминале никогда не выполняйте `source /opt/ros/jazzy/setup.bash`.** ROS 2 принесёт свою
CycloneDDS, и связь сломается. Платформа этого не проверяет — это правило окружения
([ADR-0022](decisions/0022-vendor-dds-environment.md)).

```bash
CYCLONEDDS_HOME=$HOME/g1/cyclonedds-install pip install --no-cache-dir "cyclonedds==0.10.2" numpy pyyaml
pip install --no-cache-dir --no-deps -e ~/g1/unitree_sdk2_python
```

Проверка:

```bash
python -c "import cyclonedds, unitree_sdk2py.core.channel; from importlib.metadata import version; print('cyclonedds', version('cyclonedds'))"
```

Должно напечататься ровно `cyclonedds 0.10.2`. Любая другая версия — `pip uninstall cyclonedds` и
повторить строку с `CYCLONEDDS_HOME`.

## Шаг 5. Платформа и пакет G1 в тот же venv

```bash
pip install --no-cache-dir pydantic pyyaml typer packaging mcap
pip install --no-cache-dir --no-deps -e ~/heron -e ~/heron/examples/plugins/unitree_g1
```

Первая строка — настоящие зависимости ядра плюс `mcap` для записи прогона. Вторая — платформа и
пакет робота без разрешения зависимостей.

Проверка:

```bash
heron version
heron skill list                                                  # goto_joints, hold, move_joints
python -c "from heron.core.plugins import available; print(available('adapters'))"
```

В последней строке должен быть `unitree_g1_dds`. Нет — пакет не установился, вернитесь к шагу 5.

## Шаг 6. Описания робота от производителя

```bash
cd ~/heron && python examples/robots/g1_23dof/fetch_description.py
```

В репозитории их нет: хранятся источник и коммит, а не 110 МБ мешей. Без них валидатор скажет
`urdf.file_not_found`.

---

# Часть 2. Настройка под ваш экземпляр

## Шаг 7. Найти сетевой интерфейс

```bash
ip -brief address
ping -c3 <IP робота>
```

Запомните **имя интерфейса**, которым машина видит робота (например `enp2s0`), — не IP, а имя.
Домен и адресацию сверьте с документацией производителя: платформа их только передаёт.

## Шаг 8. Добавить в профиль связь с железом

В [examples/robots/g1_23dof/profile.yaml](../examples/robots/g1_23dof/profile.yaml) найдите блок
`adapters:` и допишите в конец:

```yaml
  real:
    plugin: unitree_g1_dds
    params: {domain: 0, interface: enp2s0}
```

`domain: 0` — домен настоящего робота (у двойника производителя 1). `interface` — то, что вы нашли
на шаге 7. `peers` на настоящей сети не задают: обнаружение идёт мультикастом.

Адаптеры в контракт не входят, поэтому `contract_id` не меняется: ни `revision`, ни
`contract lock` трогать не нужно.

## Шаг 9. Создать сцену

Файл `~/g1/g1_real.yaml`:

```yaml
schema_version: 1
name: g1_real_bench
world: {backend: dummy}
robots:
  - {id: g1, profile: g1_23dof, role: scripted}
task:
  skill: move_joints
  target: [0,0,0,0,0,0, 0,0,0,0,0,0, 0, -0.5,0.3,0,0.8,0, -0.5,-0.3,0,0.8,0]
  duration: 8.0
```

`world: dummy` — потому что симулятора здесь нет, робот настоящий; сценарий всё равно проверяется
целиком ([ADR-0051](decisions/0051-a-real-run-is-chosen-by-target.md)). Цель — руки вперёд и локти
согнуты, ноги в нуле, в порядке `contract.joint_order`. `duration: 8.0` — медленно: у G1 замедлять
можно только длительностью (см. шаг 13).

## Шаг 10. Проверки, пока робот выключен

```bash
cd ~/heron
heron validate ~/g1/g1_real.yaml -R ~/heron/examples/robots
heron run ~/g1/g1_real.yaml -R ~/heron/examples/robots --target real --speed-scale 1 --dry-run
```

`validate` должен дать «ошибок 0»; три предупреждения — про свободную базу и расхождение URDF с
MJCF — ожидаемы. Сухой прогон ничего не подключает и печатает:

```
мир       действительность, окружение мира не применяется
связь     real -> unitree_g1_dds
```

---

# Часть 3. Робот

## Шаг 11. Безопасность — до включения

- **Робот на подвесе.** Уровень доступа `low` — встроенный контроллер выключен, равновесие не держит
  никто, а навык командует всеми 23 суставами, включая ноги
  ([ADR-0018](decisions/0018-free-base-is-a-run-condition.md)). Стоящий на полу робот упадёт.
- **Встроенный сервис движения выключает человек** средствами производителя. Платформа не
  переключает режимы робота, не трогает питание и тормоза и не проверяет, что это сделано: такие
  действия остаются за человеком ([ADR-0054](decisions/0054-rc-body-identity-and-a-human-releases-brakes.md)).
- **Физическая кнопка останова — под рукой.**

## Шаг 12. Первое подключение — без единой команды

Самый безопасный способ проверить связь: `connect` только ждёт состояние робота и сверяет, то ли это
тело; команд он не шлёт. Сохраните как `~/g1/read_state.py`:

```python
from pathlib import Path

from heron.core.loading import load_model
from heron.core.models import RobotProfile
from unitree_g1.adapter import G1DdsAdapter

profile = load_model(Path.home() / "heron/examples/robots/g1_23dof/profile.yaml", RobotProfile)
adapter = G1DdsAdapter()
adapter.connect(profile, {"domain": 0, "interface": "enp2s0"})
try:
    state = adapter.read_state()
    print("метка времени:", state.timestamp)
    print("позы суставов:", [round(q, 3) for q in state.joint_position])
finally:
    adapter.disconnect()
```

```bash
python ~/g1/read_state.py
```

Напечатались 23 числа — связь есть, тело опознано. Робот при этом не получил ни одной команды.

## Шаг 13. Запуск

```bash
heron run ~/g1/g1_real.yaml -R ~/heron/examples/robots --target real --speed-scale 1 --record ~/g1/first.mcap
```

**Шкала скорости у G1 — только `1`.** Адаптер объявляет `supports_speed_scale = False`: поток команд
планирует навык, связь замедлить его не может
([ADR-0052](decisions/0052-speed-scale-is-a-run-quantity.md)). Первый прогон замедляется большим
`duration` в сцене. Осторожно: сухой прогон `--speed-scale 0.3` проходит, а настоящий отменяется —
проверка объявления шкалы стоит после выхода из сухого прогона.

Остановить — **Ctrl+C**: исполнитель сначала останавливает робота, потом выходит. Аварийный останов —
демпфирование (`kp = 0`, `kd` из контракта, [ADR-0028](decisions/0028-g1-emergency-stop-damps.md)):
робот обмякает. На подвесе это нормально, стоящий — упадёт. После останова адаптер отвергает все
команды до `reset()`.

---

# Частые ошибки

| Сообщение | Что не так |
|---|---|
| `heron: command not found` | не активирован venv: `source ~/g1/venv/bin/activate` |
| `env.missing … модуль Python cyclonedds` | шаг 4 сделан не в этом venv |
| `env.forbidden … cyclonedds >=0.11` | стоит новая версия: `pip uninstall cyclonedds`, затем строка с `CYCLONEDDS_HOME` из шага 4 |
| `env.missing … unitree_sdk2py` | не выполнена последняя строка шага 4 |
| `ModuleNotFoundError: pydantic` (или `typer`) | не выполнена первая строка шага 5 |
| pip ищет `heron` в интернете и падает | забыт `--no-deps` во второй строке шага 5 |
| `профиль 'g1_23dof' не найден` | неверный путь в `-R`: нужен `~/heron/examples/robots` |
| `urdf.file_not_found` | не выполнен шаг 6 |
| `нет адаптера 'real'` | не дописан блок из шага 8 |
| `цель 'real': шкала скорости обязательна` | добавьте `--speed-scale 1` |
| `адаптер 'unitree_g1_dds' не исполняет шкалу скорости` | допустима только `--speed-scale 1`; замедляйте через `duration` |
| `связь 'real' не установлена` | не тот интерфейс или домен, робот не в низкоуровневом режиме, сеть |
| `на связи тело mode_machine = 5 (29 DoF)` | у вас G1 на 29 приводов, а профиль на 23: нужен свой профиль с другим `joint_order` |
| `состояние не обновлялось …` | сторожевой таймер: робот замолчал — сеть, питание, режим |

---

# Что происходит внутри

Порядок, в котором `run` всё решает, — и почему отказ приходит до движения, а не во время
([ADR-0025](decisions/0025-environment-requirements.md), [ADR-0051](decisions/0051-a-real-run-is-chosen-by-target.md)):

1. Сценарий проверяется валидатором целиком; ошибки — прогона нет.
2. Цель `real` требует рамку `limits.joint_position` и `limits.joint_velocity` (у G1 они есть) и
   явную `--speed-scale`.
3. Навык ищется через `skills` профиля; окружение мира на `real` **не** применяется.
4. Проверяются требования адаптера к машине: `cyclonedds == 0.10.2`, запрет `>= 0.11`,
   `unitree_sdk2py`.
5. `connect`: канал CycloneDDS открывается, ждётся первый `LowState` (до 10 с), `mode_machine`
   сверяется с суставами контракта — 4 это 23 DoF, 5 это 29
   ([ADR-0027](decisions/0027-g1-motor-slots-and-machine-identity.md)). Не то тело — отказ, команд не
   было.
6. Такт 50 Гц (`command_rate_hz`): навык считает позу, адаптер кладёт её в слоты моторов
   производителя вместе с `kp`/`kd` из контракта, пустые слоты выключены (`mode 0`). Поза вне рамки
   профиля — отказ **до отправки** ([ADR-0026](decisions/0026-refused-command-stops-the-robot.md)).
7. Состояние молчит дольше `max(10 / state_rate_hz, 0.5 с)` — сторожевой таймер: сначала останов,
   потом отчёт. Метка времени состояния — часы робота (`LowState.tick`), не машины.
8. Запись `--record` пишется по ходу прогона: каждое состояние и каждая принятая команда, с
   контрактом в метаданных ([ADR-0038](decisions/0038-the-executor-records-the-run.md)). Открывается
   Foxglove.
