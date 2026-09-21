# Общая рабочая ветка experiments/buyanov

Объединены Buyanov `afbac78`, Gerasimov с экспериментом дедупликации `4bfe1db`
и Morev `b17916b`. История всех трёх сохранена merge-коммитами.

Основной рецепт — `configs/detector-native.json`: оптимизированный расчёт
Gerasimov, без дедупликации и без экспериментальных локальных 3D-поверхностей.
Morev доступен через `configs/detector-head-supported-experimental.json` и
соседние head-конфиги. Его известные отказы не исправлены самим объединением.

Диагностика Buyanov переносится через `association_diagnostics: true` в рецепте
replay. Она сохраняет события рождения/сопоставления/разрывов и добавляет
`observations` в summary. Без неё полные объекты остаются в JSONL, а runner
держит в памяти только компактную сводку; подробный отчёт доступен через
`tunnel_guard.report_observations`. Поддержан `--workers`, по умолчанию 1.
Независимые записи при `--workers 2` используют spawn; для измерения latency
нужен один процесс. Prefetch и поэтапный журнал времени сохранены.

Конфликтующие модели геометрии не смешаны. Текущий детектор использует
`geometry.py`; прежний оцениватель Buyanov сохранён как `geometry_buyanov.py`
и используется инспектором `inspect_bag`. Отчёт инспектора явно отмечает
`buyanov_legacy_inspection_not_native_detector`. Его path_horizon не следует
выдавать за поддержанную дальность текущего детектора. Для точного воспроизведения
старых результатов нужны также зависимости и исходники соответствующего commit.

Сохранены оба ROS-интерфейса:

- `python -m tunnel_guard.ros_node`: адаптер Gerasimov, параметры `config`,
  `input_reliability`, `input_timeout_s`, публикации `/perception/...`.
- `ros2 launch tunnel_guard_ros tunnel_guard.launch.py`: совместимый адаптер
  Buyanov из `ros_node_buyanov.py`, параметр `config_path`, публикации
  `/tunnel_guard/...`. Оба используют текущий Detector; launch Buyanov теперь
  выбирает native-конфиг.

Docker сохраняет colcon-упаковку Buyanov, добавляет сборку native-модуля,
заголовки, зависимости и прямой launch Gerasimov. Новый объединённый контейнер
и live ROS на Mac не запускались; прежние отчёты Docker не подтверждают эту сборку.

## Проверка реальным исполнением

Native-модуль собран заново. Выполнены:

```sh
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m tunnel_guard.run --experiment configs/experiments/buyanov-integration-native.json --workers 1
.venv/bin/python -m tunnel_guard.run --experiment configs/experiments/buyanov-integration-association.json --workers 2
.venv/bin/python -m tunnel_guard.run --experiment configs/experiments/buyanov-integration-morev.json --workers 1
.venv/bin/python -m tunnel_guard.inspect_bag data/sourcecraft_subset/for_hackathon/doubleT_obstacle --max-frames 1 --output build/buyanov-integration-20260921-inspection.json
```

Основной replay: по 60 последовательных кадров `doubleT_obstacle` и
`roundT_doubleT`. Относительно сохранённого прогона `4bfe1db` нет изменений
сравниваемых объектов, ID, статусов, геометрии, позы, одометрии, наблюдаемости,
health и supported_range при допуске 1e-10. Диагностический replay: по три кадра
тех же записей с двумя worker-процессами, решения также совпали.
Morev head-supported: по три кадра, CLI завершился успешно; это проверка
работоспособности объединения, не новая оценка качества его алгоритма.
Инспектор успешно обработал один реальный кадр старой моделью.

Сводка сравнения — `results/buyanov-integration-20260921.json`; полные исходники,
конфиги, manifests и JSONL — `build/buyanov-integration-20260921-*`.
Конец основного replay пересёкся с диагностическим запуском, поэтому времена
этих прогонов не используются как доказательство производительности.
Автотесты не создавались и не запускались. Цель 100 мс/кадр остаётся открытой.
