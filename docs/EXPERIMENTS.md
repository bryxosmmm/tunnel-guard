# Реальные прогоны аудита — 2026-09-15

## Протокол

Сравнивались **первые 30 последовательных кадров** `doubleT_obstacle` и `doubleT_platform`, без пропуска (`every=1`), seed `20260915`. Это примерно 2.9 с каждого проезда, не полные записи. До редактирования сохранены исходники/конфиги исходной ревизии в `build/audit-baseline/source/`. Каждый runner сохраняет исходники, конфиг, environment, JSONL, metadata hash и описание файлов данных. Сводные доказательства: [`results/audit-20260915.json`](../results/audit-20260915.json).

Python 3.13.5, macOS arm64; зависимости установлены из uv.lock. Это не замер на целевом Intel и не realtime deployment. Автоматические тесты и синтетические сценарии не запускались согласно AGENTS.md.

## Данные

Архив организаторов найден в `archive/for_hackathon.zst`. Извлечены только два проезда (SQLite3), остальные четыре не обрабатывались. В bags соответственно 201 и 345 сообщений, но проверен только префикс.

| Свойство первого скана | doubleT_obstacle | doubleT_platform |
|---|---:|---:|
| PointCloud2 slots | 921600 | 307200 |
| Валидные XYZ | 346808 | 185638 |
| Невалидные/нулевые XYZ | 574792 | 121562 |
| frame_id | lidar_livox | hesai_lidar |
| rings | 128 | 128 |
| Уникальные point timestamps | 3600 | 1200 |
| Полный временной размах slots, мс | 99.973 | 33.306 |
| Размах **валидных** точек, мс | 68.057 | 32.056 |
| Медиана периода header, мс | 100.007 | 100.001 |

Поля обоих: x/y/z/intensity FLOAT32, ring UINT16, timestamp FLOAT64. Timestamp начинается с offset 18; чтение как packed xyz без учёта layout было бы ошибочным. Header совпадает с минимальным point timestamp; bag record time находится в совершенно другой абсолютной эпохе. Его нельзя использовать как возраст сенсорного измерения. В этих bags нет отдельных IMU, odometry и TF topics. Наличие ring и timestamp не доказывает калибровку или отсутствие предыдущего deskew.

Первый obstacle-скан: по радиальной дальности 100–150 м — 776 валидных точек, 150–220 м — 209; platform — 241 и 281. Это все видимые поверхности, не поддержка препятствий. Нельзя выводить дальность обнаружения. В detector output новые `returns_before_geometry_voxel` и `returns` используют **forward x** после пространственного crop; это другой знаменатель, чем радиальный histogram инспектора.

Полные схемы/префиксы диагностики: `build/audit-input-obstacle.json`, `build/audit-input-platform.json`; дополнительные времена — `build/audit-point-clocks.json`.

## Результаты и ablation

| Режим | obstacle: obstacle/unresolved | platform: obstacle/unresolved | p50 обработки, мс (obstacle/platform) |
|---|---:|---:|---:|
| Исходная ревизия, часы bag, deskew включён | 30/0 | 28/2 | 749 / 526 |
| Текущий код, часы acquisition, deskew включён | 30/0 | 28/2 | 715 / 498 |
| Текущий код, deskew выключен, headless | 30/0 | 29/1 | 753 / 528 |
| Финальный код, deskew выключен, запись RViz | 30/0 | 29/1 | 703 / 492 |

Все статусы — **выходы детектора**, а не ground truth и не число ложных тревог. Замена часов не изменила статусы префикса, но повлияла на отдельные временные подтверждения. Отключение deskew изменило один статус platform и наборы объектов. Рост/снижение реального качества без exhaustive labels не установлен. Отсутствие оптимизации и последовательные короткие замеры не позволяют назвать разницу времени ускорением.

Для финального прогона p95 обработки: 784/589 мс; wall time всей обработки с записью replay: 25.55/18.12 с. При периоде входа 100 мс текущий алгоритм **не работает в реальном времени**. `processing_s` — ядро, `read_and_process_s` добавляет чтение, `visualization_s` — отдельно сериализация/запись. `summary.visualization_total_s` и `wall_s` учитывают стоимость демонстрации; физическая end-to-end задержка live ROS не измерялась.

### Локализация

Исходные пять аннотаций не менялись. Создан только явно ограниченный evaluation view `build/audit-prefix-annotations.json`, содержащий frame 0. В baseline и финальном прогоне одна и та же предварительная структура сопоставилась при IoU ≥0.25: **IoU 0.280526**, ошибка forward-min distance **0.199038 м**. Нет exhaustive negative frames, precision=null. Это не оценка общей полноты и не повторное подтверждение прежних 5/5. Остальные четыре аннотированных кадра находятся за пределами префикса.

### Независимость display и временное подтверждение

- На 60 кадрах совпали status, track_id, confirmed, hits, confirmation, bbox, path_relation и distance между режимами display/headless; повторно сопоставлен финальный прогон.
- Побитно pose/covariance не совпали: максимальная разность элемента pose около 3.7×10⁻¹⁵. Численная идентичность KISS при многопоточном исполнении не заявляется.
- При ручной повторной подаче первого **реального** скана `Detector.process` выдал ValueError, frame_number остался 1. После второго нового скана frame_number=2, максимальный hits=2. Артефакт: `build/audit-manual-repeat.json`.
- Транспортные дубли, time reversal, смена frame_id, gap, исчезновение и malformed clouds не внедрялись в искусственные bags. Их защитные ветки проверены чтением кода; успешные динамические проверки этих сценариев не заявляются.

### Визуализация

`build/audit-reviewed/{doubleT_obstacle,doubleT_platform}_rviz/`: по 30 PointCloud2, MarkerArray и status сообщений, всего 90 на проезд. CDR сериализация и обратное чтение настоящих сообщений выполнены; headers совпадают с временем записи результатов, в каждом кадре есть DELETEALL и пометка REPLAY. Не обнаружены нечисловые координаты показанного облака. `candidate_measurements` сохраняет реальные точки-представители кандидатов отдельно от прореженного display cloud.

Статический обзор реального первого кадра с увеличением: `build/audit-preview.png`. Он отрисован из прочитанных result messages и визуально просмотрен. Это отдельный PNG, **не скриншот работающего RViz**. Рецепт локального рендера сохранён в `build/render-audit-preview.py` (Pillow/NumPy из bundled runtime, не runtime-зависимости детектора).

## Выполненные команды

Все пути ниже относительно корня проекта, кроме `$PWD`, который фиксирует корень для запуска сохранённого baseline.

```sh
git status --short
git rev-parse HEAD
git diff --check
uv sync --locked
# В сети были неудачные попытки/таймауты; успешный повтор:
UV_HTTP_TIMEOUT=180 uv sync --locked

mkdir -p data/sourcecraft_subset
tar --zstd -tf archive/for_hackathon.zst
tar --zstd -xf archive/for_hackathon.zst -C data/sourcecraft_subset \
  for_hackathon/doubleT_obstacle for_hackathon/doubleT_platform

.venv/bin/python -m tunnel_guard.inspect_bag \
  data/sourcecraft_subset/for_hackathon/doubleT_obstacle \
  --max-frames 30 --output build/audit-input-obstacle.json
.venv/bin/python -m tunnel_guard.inspect_bag \
  data/sourcecraft_subset/for_hackathon/doubleT_platform \
  --max-frames 30 --output build/audit-input-platform.json

# Сохранённый baseline запускался из build/audit-baseline/source,
# абсолютные пути к исходным bags были записаны в его experiment JSON:
project_root="$PWD"
(cd build/audit-baseline/source && \
 "$project_root/.venv/bin/python" -m tunnel_guard.run \
 --experiment configs/evaluation-audit-baseline.json)

.venv/bin/python -m tunnel_guard.run --experiment configs/evaluation-audit.json
.venv/bin/python -m tunnel_guard.run --experiment build/audit-headless.json
.venv/bin/python -m tunnel_guard.run --experiment build/audit-clock-only.json
.venv/bin/python -m tunnel_guard.evaluate --run build/audit-baseline-real \
  --annotations build/audit-prefix-annotations.json \
  --output build/audit-baseline-real/localization-prefix.json
.venv/bin/python -m tunnel_guard.evaluate --run build/audit-reviewed \
  --annotations build/audit-prefix-annotations.json \
  --output build/audit-reviewed/localization-prefix.json

docker info --format '{{.ServerVersion}}'
```

До нахождения/распаковки архива запуск `evaluation-quality.json` и сохранённого baseline завершался FileNotFoundError — ранние неудачные логи сохранены. Docker info сообщил отсутствие daemon/socket; `ros2` не найден. GUI/Ubuntu/Humble, Docker build, четыре остальных проезда, полный исходный panel, synthetic stress, optional published backends и видео не проверялись. После появления данных короткий baseline и текущая итерация действительно выполнены, а не заменены синтетикой.

Одноразовые read-only Python-команды дополнительно использовались для просмотра timestamps/rings, CDR readback, сравнения JSONL и ручной подачи того же реального скана. Их численные выводы включены в `results/audit-20260915.json`. Автоматического pass/fail test runner нет.

**Не перезаписывайте результаты повторным запуском.** Runner требует новый output directory: скопируйте experiment JSON и измените `output`. Сравнивайте source snapshots и detector.json, поскольку исторические README-результаты получены до отключения deskew.

# Параллельный запуск — 2026-09-16

`tunnel_guard.run` и `tunnel_guard.stress` принимают `--workers` (по умолчанию `cpu_count`). Единица работы — независимый проезд либо независимая сцена; порядок строк в JSONL, записи `summary.json`/`manifest.json` и все агрегаты сохраняются, поэтому артефакты сопоставимы с последовательными прогонами. Каждая сцена стрессинга имеет собственный `SeedSequence([plan seed, index])`, а KISS-ICP и треки не переносят состояние между проездами, так что распараллеливание не меняет вход для детектора.

Измерения на Apple M4 (10 логических ядер), threads per process не настраивались:

| прогон | последовательно | параллельно | ускорение |
|---|---|---|---|
| synthetic `configs/stress-quality.json`, 110 сцен / 330 кадров | 97.9 с | 21.5 с (`--workers 10`) | 4.6× |
| реальные bags, 3 проезда × 15 кадров | 24.1 с | 12.7 с (`--workers 3`) | 1.9× |
| полный реальный panel, 6 проездов / 2488 кадров | 1096 с (сохранённый прогон) | ~380–450 с (оценка: критический путь — самый длинный проезд) | ~2.5× |

Проверка эквивалентности: параллельный и последовательный прогоны сравнивались по JSONL при исключении полей времени (`processing_s`, `motion_s`, `read_and_process_s`, `visualization_s`) — совпадение по статусам, числу объектов, решениям (`confirmed`, `path_relation`) и по `summary.json`.

**Панель не бит-воспроизводима, и это не регрессия.** Два прогона одного и того же кода с одним и тем же seed дают расхождение в траектории на уровне 1e-16…1e-15 и могут добавить или потерять один `adjacent`-кластер (не hazard). Код из `HEAD` и рефакторинг 2026-09-16 демонстрируют одинаковый разброс на одних и тех же сценах (39, 42, 60, 70, 84), а все агрегированные метрики (event recall, precision, F1, negative episode rate, per-range recall) совпали во всех прогонах. Источник — не KISS-ICP threads: `odometry_threads=1` снимает часть дрожания позы, но не устраняет переключения кластеров. Не публикуйте единичные per-frame `objects` как воспроизводимый результат; сравнивайте агрегаты и решения.
