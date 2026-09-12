# Heron

Платформа моделирования гетерогенных робототехнических комплексов. Один
декларативный профиль робота → генерация сред (URDF/SDF/USD, launch, compose) →
навык кодом или обученная политика за одним интерфейсом → проверка в симуляции →
перенос на железо.

Новизна сосредоточена в **валидаторе**: рассогласование сред — порядок суставов,
единицы, частоты — ловится до запуска, а не проявляется на роботе как поведение
без единой записи в логах.

Архитектура — [ARCHITECTURE.md](ARCHITECTURE.md), источник правды. Обоснование
решений — [docs/decisions/](docs/decisions/). Ход работы — [docs/journal/](docs/journal/).

## Состояние

Фаза 2 дорожной карты: из профиля и сценария порождаются среды. Работает
цепочка «профиль → валидатор → генераторы → мир Gazebo с визуализацией в
RViz». Артефакты собираются и проверяются без симулятора и без ROS 2; сам
запуск на Linux-машине ещё не выполнялся, и до него фаза не закрыта.

Навык, двигающий робота, обучение и координатор — фазы 3 и дальше.

## Установка ядра

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Ядро ставится без вендорских зависимостей: ни симулятора, ни SDK робота, ни
фреймворка обучения в обязательных зависимостях нет. Для работы с профилями,
валидатором и CLI окружение фазы 0 не требуется.

## Окружение: фаза 0

Разворачивается на Linux-машине и поднимает три мира: Gazebo Harmonic как мир
комплекса, MuJoCo как второй физический движок, Isaac Lab как среду обучения.
Всё делает [scripts/phase0.sh](scripts/phase0.sh).

Требования: Ubuntu 24.04, x86_64, GPU NVIDIA. Версия ОС жёсткая — ROS 2 Jazzy
собирается только под noble, и это же зафиксировано в
[ADR-0005](docs/decisions/0005-training-resources.md).

```bash
./scripts/phase0.sh check      # промер, ничего не меняет
./scripts/phase0.sh install    # все шаги подряд
./scripts/phase0.sh verify     # каждая среда запускается по-настоящему
./scripts/phase0.sh report     # версии сред таблицей для карточки эксперимента
```

Шаги ставятся и по одному: `./scripts/phase0.sh install ros2 gazebo`.

| Шаг | Что ставит |
|---|---|
| `base` | Пакеты сборки, `vulkan-tools`, `uv` |
| `ros2` | ROS 2 Jazzy desktop, `ros2_control`, `xacro`, MCAP-хранилище rosbag2 ([ADR-0003](docs/decisions/0003-trajectory-format.md)) |
| `gazebo` | Gazebo Harmonic, мосты `ros_gz` и `gz_ros2_control` |
| `exec` | `.venv` в репозитории: ядро с группами `sim` и `dev` — MuJoCo и pytest |
| `isaac` | `~/train-stack`: отдельный venv, Isaac Sim, Isaac Lab, rsl_rl |
| `docker` | Docker CE, compose, NVIDIA Container Toolkit |

Каждый шаг идемпотентен: повторный запуск ничего не переставляет. Проверка
места считает только то, что предстоит поставить, поэтому на машине с уже
собранным стеком она не мешает.

**Два окружения Python, а не одно.** Стек обучения живёт в `~/train-stack`
(меняется переменной `TRAIN_ROOT`), ядро и MuJoCo — в `.venv` репозитория. Это
не удобство, а инвариант 2: миры соприкасаются файлами, и ROS 2 не может
просочиться в обучение, потому что не делит с ним интерпретатор.

### Что делает человек, а не скрипт

Скрипт ничего не чинит молча: нарушенное предусловие останавливает работу и
печатает команду, которую выполняют руками.

- **Драйвер NVIDIA** — `sudo ubuntu-drivers install` и перезагрузка. Ставится
  отдельно: его версия входит в условия каждого эксперимента (раздел 13).
- **Подкачка не меньше 16 ГиБ** при 15 ГиБ RAM ([ADR-0005](docs/decisions/0005-training-resources.md)).
  Переразбивать диск не нужно, достаточно файла на корне:

  ```bash
  sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```

- **Лицензия NVIDIA** — шаг `isaac` идёт только с флагом:

  ```bash
  ./scripts/phase0.sh install isaac --accept-nvidia-eula
  ```

### Особенности, на которые наступали

- `OMNI_KIT_ACCEPT_EULA=YES` — переменная окружения, а не аргумент: ставится
  **перед** командой. Для проверки её выставляет сам скрипт.
- Первый запуск Isaac Sim компилирует шейдеры — десятки минут тишины, похожей
  на зависание.
- `isaaclab.sh` ставит модули по очереди и **завершается успешно, даже когда
  один из них не собрался**. Скрипт проверяет набор пакетов после установки,
  иначе пропажа всплывает позже — импортом внутри тьюториала.
- `flatdict` ставится без изоляции сборки: его `setup.py` импортирует
  `pkg_resources`, которого нет в setuptools 81 и новее. Без этого не
  собирается и сам пакет `isaaclab`.
- Isaac Lab поднимает `psutil` и `typing_extensions` выше пинов
  `isaacsim-kernel`. Скрипт печатает конфликты предупреждением: существенны они
  или нет, показывает только запуск `verify isaac`.

## Что можно сделать сейчас

```bash
heron capabilities                        # реестр возможностей ядра
heron profile show examples/robots/arm6.yaml
heron contract id examples/robots/arm6.yaml
heron validate examples/robots/*.yaml examples/scenarios/table.yaml -R examples/robots
heron profile new my_robot --into examples/robots
heron skill check hold examples/robots/rest_arm.yaml
```

Последняя команда — самая показательная: навык требует потокового управления,
робот принимает команды целиком, и это ошибка валидатора, а не сюрприз во время
движения.

## Среды из профиля

```bash
heron gen examples/scenarios/table.yaml --into generated/table -R examples/robots
heron gen examples/scenarios/table.yaml --into generated/table -R examples/robots --check
heron sim run examples/scenarios/table.yaml --into generated/table -R examples/robots --dry-run
```

Порождается то, что выводимо из профиля и сценария:

| Файл | Из чего выведен |
|---|---|
| `<робот>/description.urdf` | входное описание плюс блок `ros2_control` из контракта |
| `<робот>/controllers.yaml` | тип действия, режим управления, `state_rate_hz`, `joint_order` |
| `world.sdf` | сценарий; шаг физики — из наибольшей частоты состояния |
| `bringup.launch.xml` | состав сценария, позы, идентификаторы роботов |
| `view.rviz` | корневые звенья описаний и префиксы систем отсчёта |
| `manifest.json` | чем порождено, из чего и при каком `contract_id` |

**Кинематику платформа не сочиняет.** `description.urdf` в профиле и
`world.file` в сценарии — вход: генератор их не перезаписывает, а валидатор
сверяет описание с профилем ([ADR-0009](docs/decisions/0009-generators-do-not-author-kinematics.md)).
Отсюда правила `urdf.*`: сустав, переименованный в модели и не переименованный в
профиле; сустав, который контракт называет управляемым, а модель заваривает;
лимит профиля шире физического предела. Единицы при сверке переводятся — URDF по
спецификации в радианах, а контракт может объявлять градусы.

Генерация детерминирована, поэтому `--check` отвечает на вопрос «артефакты
отстали от профиля?» без запуска чего бы то ни было
([ADR-0010](docs/decisions/0010-generated-artifacts.md)). Выходной каталог не
версионируется: производную величину рядом с исходной не хранят.

Запуск сцены целиком — `heron sim run` без `--dry-run`; он требует Linux с ROS 2
Jazzy и Gazebo Harmonic, то есть окружения фазы 0. Сцену, не прошедшую валидатор,
бэкенд не строит вовсе ([ADR-0011](docs/decisions/0011-sim-backend-runs-processes.md)).

## Проверка

```bash
pytest
ruff check src tests
mypy
```

Отдельная проверка инварианта 1 — ядро не знает имён роботов:

```bash
pytest tests/test_invariants.py
```

## Лицензия

MIT.
