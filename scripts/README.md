# scripts/

Что здесь лежит, кто за что отвечает и в каком порядке запускать. Устройство —
[ADR-0071](../docs/decisions/0071-one-installer-packages-install-themselves.md), подробности установки —
[docs/install.md](../docs/install.md).

```
scripts/
├── install.sh          установка и проверка Linux-машины — единственный вход
├── activate.sh         войти в окружение: source scripts/activate.sh <exec|ros|train|пакет>
├── dev-setup.ps1       Windows: окружение разработки, как в CI
├── lib/common.sh       общие функции и пути (подключается скриптами, сам не запускается)
├── tools/              инструменты: запускаются руками или из прогонов
└── runs/               приёмочные прогоны на Linux, ещё не пройденные
```

## Установка

| Скрипт | Где | Что делает |
|---|---|---|
| [`install.sh`](install.sh) | Linux | Ставит платформу (`base ros2 gazebo exec descriptions train docker`) и окружения пакетов роботов: находит `examples/plugins/*/install.sh` и запускает их. Команды `install`, `check`, `verify`, `sync`, `status`, `report`; справка — `scripts/install.sh help` |
| [`activate.sh`](activate.sh) | Linux | Подключает в текущую оболочку окружение под задачу: `exec` (`.venv`), `ros` (ROS 2, рабочие пространства пакетов, `.venv`), `train`, `<пакет>` (venv пакета робота). Отказывает, если ROS 2 подключён там, где ему не место |
| [`dev-setup.ps1`](dev-setup.ps1) | Windows | `.venv` на Python 3.11 с группами `dev record onnx testing` и всеми пакетами `examples/plugins`; кэши на D. `-Descriptions` — ещё и описания производителей |
| [`lib/common.sh`](lib/common.sh) | — | Пути (`.venv`, `~/train-stack`, `~/robot-stack`), имя платформы из `pyproject.toml`, вывод, `apt`, загрузка исходника на закреплённом коммите |

Установщики пакетов лежат не здесь, а в самих пакетах: окружение пакета объявляет и строит пакет.

| Установщик | Части |
|---|---|
| [`examples/plugins/unitree_g1/install.sh`](../examples/plugins/unitree_g1/install.sh) | `dds` — CycloneDDS 0.10.2; `venv` — окружение адаптера; `ws` — `unitree_hg` и мост Gazebo; `twin` — двойник MuJoCo |
| [`examples/plugins/robopro_rc/install.sh`](../examples/plugins/robopro_rc/install.sh) | `venv` — окружение стенда с API производителя (`API_DIR`) |

## Инструменты — `tools/`

| Скрипт | В каком окружении | Что делает |
|---|---|---|
| [`gains-check.py`](tools/gains-check.py) | `exec` | Устойчивы ли `kp`/`kd` профиля при шаге `1/state_rate_hz` на инерциях описания; код 1 — сустав неустойчив (ADR-0045, ADR-0063). `python scripts/tools/gains-check.py <profile.yaml>` |
| [`arm-gains.py`](tools/arm-gains.py) | `exec` | Выводит `kp`/`kd` для последовательной руки по инерциям описания (так получены коэффициенты `arm6`) |
| [`record-metrics.py`](tools/record-metrics.py) | `exec` | Числа прогона из записи MCAP, читая её библиотекой `mcap`, а не читателем платформы; `run` — одна запись, `table` — сводная таблица |
| [`onnx-compare.py`](tools/onnx-compare.py) | `exec` и `train` | ONNX политики снаружи: `describe` — входы и выходы, `torch` — эталон TorchScript, `compare` — расхождение сетей на одних входах |
| [`isaac-urdf-probe.py`](tools/isaac-urdf-probe.py) | `train`, через `isaaclab.sh -p` | Что Isaac Lab делает с описанием: порядок DOF, слитые звенья, пределы, положения тел после шагов |
| [`rc-probe.py`](tools/rc-probe.py) | `robopro_rc` | Проба контроллера RC только на чтение: метки, кнопка аварийного останова, модель (`--full`), пары поз для сверки кинематики |

## Приёмочные прогоны — `runs/`

Каждый прогон — шаги по имени (`bash scripts/runs/<прогон>.sh env plan`), журнал в своём каталоге,
таблица итогов в конце; присылать — файл журнала целиком. Шаг `env` обновляет нужные окружения
(`scripts/install.sh sync …`), ничего не собирая.

| Прогон | Этап | Что доказывает | Нужно поставить |
|---|---|---|---|
| [`rc5-stand.sh`](runs/rc5-stand.sh) | фаза 5, этап 4 | RC5 на стенде от безопасного к рискованному: проба, отказы, движение, останов, задача | `robopro_rc` |
| [`arm6-train.sh`](runs/arm6-train.sh) | фаза 6, этап 4 | Обучение `arm6` в среде, выведенной из контракта; порог ADR-0005 снаружи; артефакт | `exec train` |
| [`arm6-learned-gazebo.sh`](runs/arm6-learned-gazebo.sh) | фаза 6, этап 5 | Обученный навык в Gazebo тем же исполнителем; замена навыка — одна строка профиля | `exec`, итог `arm6-train.sh` |
| [`g1-upper-learned.sh`](runs/g1-upper-learned.sh) | фаза 6, этап 7 | Весь конвейер на закреплённом G1: крепление в Isaac, обучение, Gazebo через мост | `exec train unitree_g1` |
| [`complex-mission.sh`](runs/complex-mission.sh) | фаза 8, этап 3 | Одна миссия, два робота, один мир Gazebo; останов комплекса на отказе | `exec unitree_g1` |

Прогон, который пройден и закрыт фазой, удаляется: его версия остаётся в истории git, результат — в
журнале и карточке эксперимента.

## Удалённые скрипты

До ADR-0071 здесь лежали скрипты по этапам. Их можно найти в истории (`git log --all -- scripts/<имя>`):

| Было | Стало |
|---|---|
| `phase0.sh` | `install.sh` (компоненты платформы) |
| `phase4-probe.sh` | удалён: факты 0.5–0.8 закрыты (ADR-0022); сборка CycloneDDS и venv — `unitree_g1/install.sh`, части `dds`, `venv` |
| `phase4-stage1.sh` | удалён: этап закрыт; сборка двойника — часть `twin` |
| `phase4-stage4.sh`, `phase4-stage5.sh`, `phase4-stage7.sh` | удалены: фаза 4 закрыта (0.4.0); сборка моста — часть `ws`, сцена комплекса — `runs/complex-mission.sh` |
| `phase4-stage6.sh` | удалён: эксперимент 0001 проведён, его числа — в карточке |
| `phase4-stage6-metrics.py` | `tools/record-metrics.py` |
| `phase5-probe.py`, `phase5-stage4.sh` | `tools/rc-probe.py`, `runs/rc5-stand.sh` |
| `phase6-probe.sh` | удалён: факты 0.13–0.18 закрыты, эксперимент 0002 проведён |
| `phase6-probe-onnx.py`, `phase6-probe-urdf.py` | `tools/onnx-compare.py`, `tools/isaac-urdf-probe.py` |
| `phase6-arm6-gains.py`, `phase6-gains-check.py` | `tools/arm-gains.py`, `tools/gains-check.py` |
| `phase6-stage4.sh`, `phase6-stage5.sh`, `phase6-stage7.sh` | `runs/arm6-train.sh`, `runs/arm6-learned-gazebo.sh`, `runs/g1-upper-learned.sh` |
| `phase8-stage3.sh` | `runs/complex-mission.sh` |
