# Итерация Buyanov: наблюдаемость и диагностика ID, 2026-09-20

Работа в `experiments/buyanov`, база — обновлённый `origin/main`, `592814e`.
`main` и история других веток не изменялись. Автотесты и subagents не использовались.

## Изменение

Гипотеза: причины разрывов сопровождения и пределы геометрии можно сделать
проверяемыми без изменения порогов, association или решений детектора.
Критерий — одинаковые решения на фиксированном реальном префиксе и отдельные
измеримые отчёты для старых bags, нового unlabeled источника и synthetic panel.

- `inspect_bag` сохраняет ограничение покрытия, layout каждого сообщения,
  acquisition/record clocks, частоту, дубли/обратный ход/разрывы времени, геометрию,
  остатки подгонки, колею, высоты головок и неопределённость по дальности.
  Исходники, конфигурация, среда, revision/status/patch лежат рядом в `.source/`.
- `TrackGeometry.describe()` дополняется наблюдениями по каждой стороне рельсовой
  пары и профилем uncertainty. Измерения берутся около уже выбранных центров и
  не возвращаются в fitting или acceptance.
- `association_diagnostics: true` в experiment включает анализ реального прогноза
  **до** Kalman update. Для рождения берётся ближайший активный предшественник;
  для неназначенного трека — ближайшее текущее наблюдение; отдельно сохраняется
  возобновление прежнего ID после отсутствовавших кадров. Gates не расширяются.
- Для каждой пары сохраняются ID, кадр, acquisition timestamp, spatial/temporal gap,
  Mahalanobis squared, extent distance, причины отклонения, predicted/innovation
  sigma, bbox IoU и двусторонний overlap поддержки в оценённой мировой системе.
  Допуск overlap равен существующему `cluster_voxel_m`, 0,05 м. Старая поддержка
  переносится к предсказанному центру без моделирования вращения объекта.
- `report_observations` суммирует реальные JSONL и сравнивает общие `(bag, frame)`.
  Он не присваивает identity ground truth и не оценивает real precision/recall.
  Диагностика имеет вычислительную стоимость; вывод времени отделён от решений.

Названия `split`, `merge`, `fragmentation_or_occlusion`, `occlusion_or_missed_segmentation`
— **гипотезы**, выводимые из назначения и overlap, не доказанные причины. Ближайший
сосед может быть другим объектом. Одна пара может повторяться в нескольких кадрах;
`different_id_hypotheses` считает наблюдения пар, не физические ID switches.
`verified_identity_switches = null`. Сброс при недостоверной позе или временном
разрыве фиксируется отдельно; сравнение точек через такой сброс не выдумывается.

## Семантика статусов

Совместимый frame status `obstacle` сохранён. В сводке его явное имя —
`confirmed_obstacle`: алгоритм подтвердил пересечение **reference contour**, а не
фактическую коллизию. Object-level сводка различает:

| Состояние | Основание |
|---|---|
| `observed_structure` | Подтверждённая поддержка, `path_relation=adjacent` |
| `candidate` | Недостаточно подтверждения |
| `unresolved_obstacle` | Подтверждённая номинальная интрузия при недостаточной геометрии |
| `confirmed_obstacle` | Подтверждённое алгоритмом поддержанное пересечение |
| `unknown` | Frame: недостаточно геометрии/возвратов |
| `no_obstacle_observed` | Frame: опасность не зарегистрирована; путь не объявляется свободным |

## Покрытие и источник

Старый набор: все шесть `data/sourcecraft_subset/for_hackathon/*`, первые 30 кадров
каждого, `every=1`. Это 180 наблюдений и около 2,9 с каждого проезда.
Частичная историческая разметка не превращена в exhaustive labels.

Архив `/Users/pbuyanov/Downloads/датасет/new_data.zst` проверен потоковым чтением
**всего tar** без извлечения: 221 SQLite, 11 271 сообщений, 90 121 539 170 байт
регулярных файлов. Изначально свободно около 25 GiB; полный архив не помещается.
Извлечены первые 20 сегментов в отдельный новый `data/new_data/`, 1 020 сообщений,
source frames 0–1019. Это **одна непрерывная запись**, не 20 экспериментов.
Остальные 201 сегмент не извлечены. Полные 20 минут не проверены.

`data/new_data/source-metadata.yaml` — неизменённая исходная metadata;
`metadata.yaml` — производная с точным списком, числом и длительностью извлечённых
сегментов. `extraction.json` сохраняет границы покрытия и SHA256 всех 20 файлов.
Копия manifest и скрипт извлечения сохранены в `build/buyanov-20260920-evidence/`.
Manifest каждого нового runner/inspector включает сведения об извлечении.
Labels и совместимость происхождения со старым набором неизвестны. Alarm counts
на новом наборе — **unlabeled observations**, не false positives.

## Sensor и geometry inspection

По 30 реальных сообщений каждого источника, сначала шесть старых bags, затем новый.
Везде layout `x/y/z/intensity FLOAT32`, `ring UINT16`, `timestamp FLOAT64`,
`point_step=26`; offset timestamp = 18. Полей return нет. Частота около 10 Гц.
`doubleT_obstacle`: `lidar_livox`, 921 600 slots; остальные источники:
`hesai_lidar`, 307 200 slots. Это имена, записанные producer, а не подтверждённые
модель сенсора или оси установки. Slots не равны валидным или независимым измерениям.

| Источник, первые 30 | Median inner-gauge proxy, м | Median path horizon, м |
|---|---:|---:|
| doubleT_obstacle | 1,4854 | 57 |
| doubleT_platform | 1,4753 | 59,5 |
| roundT_doubleT | 1,4731 | 57 |
| roundT_pressureGate_roundT | 1,4741 | 62 |
| roundT_squareT_pressureGate_squareT | 1,4678 | 57 |
| squareT_platform_squareT_switch | 1,4737 | 57 |
| new_data | 1,4622 | 57 |

Геометрия валидна по внутренним критериям на всех 210 inspected кадрах. Эти числа
получены после настроенного transform, range/crop/voxel без ICP/deskew/background
rejection. В детекторе остаётся штатный полный pipeline.

Колея — оценка по видимой полосе возвратов; центры используют предположение о
ширине головки 0,07 м. Высоты — p80 точек над fitted bed, не съёмка railhead.
Lateral residual относится к выбранной полосе и предполагаемому центру, а не к
независимой истинной оси. Ground residual относится только к inliers плоскости.
Отдельная калибровка канта не выполнена. Uncertainty эвристическая, не safety bound.
`path_horizon_m` в существующей реализации учитывает латеральный путь; для
достоверности контура нужна также ground uncertainty. Новые отчёты показывают обе.
Отсутствующая поддержка обозначается `null`/unsupported, не нулевой ошибкой.

## Воспроизведение

Все outputs созданы заново; для повтора сначала скопировать recipe и выбрать новый
output. Python 3.12.14, macOS arm64, 10 logical CPUs; точные версии пакетов и hashes
исходников — в manifests. Однопроцессный запуск: `--workers 1`.

```sh
export UV_CACHE_DIR=/private/tmp/tunnel-guard-uv-cache
uv run --locked python -m tunnel_guard.run --experiment configs/evaluation-buyanov-baseline.json --workers 1
# Повторить inspect_bag для каждого из шести старых bag:
uv run --locked python -m tunnel_guard.inspect_bag data/sourcecraft_subset/for_hackathon/doubleT_obstacle --max-frames 30 --output build/buyanov-20260920-inspection/doubleT_obstacle.json
uv run --locked python -m tunnel_guard.inspect_bag data/new_data --max-frames 30 --output build/buyanov-20260920-inspection/new_data.json
PYTHONPATH="$PWD/build/buyanov-20260920-old-baseline/source" .venv/bin/python -P -m tunnel_guard.run --experiment configs/evaluation-new-data-baseline.json --workers 1
uv run --locked python -m tunnel_guard.run --experiment configs/evaluation-new-data.json --workers 1
uv run --locked python -m tunnel_guard.run --experiment configs/evaluation-buyanov-diagnostics.json --workers 1
PYTHONPATH="$PWD/build/buyanov-20260920-old-baseline/source" .venv/bin/python -P -m tunnel_guard.stress --experiment configs/stress-buyanov-baseline.json --workers 1
uv run --locked python -m tunnel_guard.stress --experiment configs/stress-buyanov.json --workers 1
```

Baseline старого набора выполнен до редактирования исходников. Для последующих
baseline используется его точный source snapshot через `python -P`/`PYTHONPATH`,
без изменения checkout. Synthetic quality: неизменённая матрица 110 сцен,
330 кадров, тот же seed и IoU 0,1; это configured evaluation, не test suite.
Неудавшийся первый `uv` запуск (недоступный cache) сохранён в `old-baseline.log`;
повтор с writable cache — в `old-baseline-run.log`. Промежуточный detector run
`old-diagnostics` также сохранён, финальный — `old-final`.

Отдельная сводка и сравнение сохранённых результатов:

```sh
uv run --locked python -m tunnel_guard.report_observations --run build/buyanov-20260920-old-final --compare-to build/buyanov-20260920-old-baseline --output build/buyanov-20260920-evidence/old-final-comparison.json
uv run --locked python -m tunnel_guard.report_observations --run build/buyanov-20260920-new-diagnostics --compare-to build/buyanov-20260920-new-baseline --output build/buyanov-20260920-evidence/new-final-comparison-v2.json
uv run --locked python -m tunnel_guard.report_observations --run build/buyanov-20260920-synthetic --compare-to build/buyanov-20260920-synthetic-baseline --output build/buyanov-20260920-evidence/synthetic-final-comparison.json
git diff --check
git status --short
```

## Конкретный разрыв ID для разбора

`doubleT_platform`, кадры 0 → 1: ID 1 → 169, dt 0,100 с. Пространственное
расстояние от предсказанного центра 1,879 м; Mahalanobis squared 15,913;
назначение отклонено по существующему Mahalanobis gate. Bbox IoU 0,538,
overlap поддержки 30,8% / 43,0% при допуске 0,05 м.

Длина **наблюдаемой** компоненты по x сократилась с 10,284 до 6,394 м;
число support voxels — с 7 950 до 5 696. Минимальный x почти неизменен
(2,001 → 2,003 м), максимальный сократился с 12,285 до 8,398 м.
Следовательно, середина bbox сместилась примерно на 1,94 м при изменении
наблюдаемой поддержки; это конкретный механизм несогласования с прогнозом
центра, не доказательство движения физического объекта. Оба наблюдения —
`adjacent`, не collision hazard. Почему изменилось членство точек в компоненте
(видимость, surface rejection, segmentation), ещё не установлено. Для дальнейшего
разбора нужны этапы поддержки, а не расширение gate.

Примеры split/merge и повторного наблюдения того же ID сохранены отдельно;
они не объявлены размеченными физическими событиями.

## Ограничения решения

Это улучшение диагностируемости, не recall или collision safety. Reference contour
не является validated vehicle swept envelope; sensor extrinsics, timing, prior
processing, колея и кант не подтверждены. Deskew и longitudinal extension остаются
выключенными, пороги и envelope неизменны. Слабые дальние кластеры остаются
проблемой. Дистанции — local sensor x поддержки, не от бампера и не вдоль пути.
Нет независимых exhaustive negative episodes: nuisance alarm rate и precision
недоступны. Кадры одного объекта и повторяющиеся hypotheses зависимы. Скорость
измерена офлайн на данной машине, без подтверждения целевого hardware/deployment.
