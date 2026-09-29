# Развёртывание Tunnel Guard по ТЗ Департамента транспорта

Эта инструкция соответствует последовательности из разделов 3–5 и 8.6 ТЗ:
`Docker build → Docker run → ROS 2 bag play → облако PointCloud2 → результат`.
Основной сценарий ниже выполняет **текущую обработку ROS 2-потока**. Отдельный
браузерный просмотрщик показывает **заранее сохранённые результаты** и не
подменяет живую обработку. Исходное ТЗ: `5. ДепТранспорта.pdf`; ограничения
реализации и точный ROS-контракт описаны в [ROS2.md](ROS2.md).

## 1. Стенд и исходные данные

Целевой стенд ТЗ: Ubuntu 22.04.5 LTS, Intel Core i7-9700E (8 ядер), 128 GiB
RAM, RTX 4070 Ti SUPER и 5,8 TB диска. Для запуска нужны Docker, исходный код
проекта и извлечённые ROS 2 bag-каталоги. ROS 2 Humble, Python-зависимости,
нативное C++-расширение и RViz2 устанавливаются **внутри образа** при сборке;
системный ROS 2 на хосте не обязателен. GPU текущий основной рецепт не использует.
Используйте проверенную интеграционную ревизию, а не старую экспериментальную
ветку: `Dockerfile` находится в корне проекта. Зафиксируйте её полный commit SHA.
При наличии доступа к приватному репозиторию:

```sh
git clone --branch main --single-branch \
  https://github.com/bryxosmmm/tunnel-guard.git
cd tunnel-guard
```

Текущий демонстрационный сервер тоже работает на Ubuntu 22.04 и восьми vCPU,
но имеет лишь 16 GiB RAM и 200 GB SSD. Его измерения нельзя выдавать за
результат на стенде ТЗ.

В каждом bag-каталоге должны быть `metadata.yaml` и `.db3`. Например:

```text
data/
└── for_hackathon/
    ├── roundT_doubleT/
    │   ├── metadata.yaml
    │   └── roundT_doubleT_0.db3
    └── doubleT_obstacle/
        ├── metadata.yaml
        └── doubleT_obstacle_0.db3
```

На уже подготовленном сервере каталог данных —
`~/tunnel-guard-deploy/data`; оба исходных архива и их SHA-256 сохранены в
`~/tunnel-guard-deploy/dataset` и `~/tunnel-guard-deploy/tunnel-guard-SHA256SUMS`.
Файл `~/tunnel-guard-deploy/logs/dataset-complete.json` подтверждает распаковку
семи bag-каталогов. На новом стенде достаточно перенести нужные **извлечённые**
bag-каталоги, сохранив `metadata.yaml` и `.db3` вместе. Для текущего сервера:

```sh
cd ~/tunnel-guard-deploy/source
DEMO_DATA_DIR="$HOME/tunnel-guard-deploy/data"
test -f "$DEMO_DATA_DIR/for_hackathon/doubleT_obstacle/metadata.yaml"
cd "$HOME/tunnel-guard-deploy/dataset" && sha256sum -c ../tunnel-guard-SHA256SUMS
cd "$HOME/tunnel-guard-deploy/source"
```

На другом стенде установите `DEMO_DATA_DIR` в абсолютный путь к извлечённому
каталогу `data`. Дальнейшие команды выполняются из корня исходного проекта.

## 2. Собрать образ

```sh
sudo docker build -t tunnel-guard:demo .
```

Сборка использует официальный ROS-репозиторий базового образа и требует сети.
Перед демонстрацией проверьте доступность образа и тему конкретного bag:

```sh
sudo docker image inspect tunnel-guard:demo --format '{{.Id}}'
sudo docker run --rm --network none \
  -v "$DEMO_DATA_DIR:/data:ro" tunnel-guard:demo \
  ros2 bag info /data/for_hackathon/doubleT_obstacle
```

У `doubleT_obstacle` входной PointCloud2 находится в
`/sensing/lidar/hesai128/pointcloud` (201 сообщений). У `roundT_doubleT` он
находится в `/lidar_points` (252 сообщения). На новом контрольном bag всегда
сначала смотрите `ros2 bag info`, а затем указывайте его фактическую тему.

## 3. Запустить живой ROS 2-поток

Терминал 1 — детектор. Устройство ROS 2 и алгоритм запускаются из образа,
а bag монтируется только на чтение. `--network host` позволяет использовать
общий DDS с другими ROS 2-процессами на этом хосте.

```sh
sudo docker run --rm -d --name tunnel-guard-live \
  --network host --ipc=host \
  -v "$DEMO_DATA_DIR:/data:ro" \
  tunnel-guard:demo \
  ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py \
  input_topic:=/sensing/lidar/hesai128/pointcloud \
  input_reliability:=reliable input_timeout_s:=3.0
sudo docker logs --tail 20 tunnel-guard-live
```

В журнале ожидается `Listening on /sensing/lidar/hesai128/pointcloud`.
Для другого bag измените `input_topic`. Если источник предлагает только
best-effort, укажите `input_reliability:=best_effort`. Подписка хранит один
последний кадр: при перегрузке промежуточные кадры могут выпадать.

Терминал 2 — результат детектора. Запустите подписку **до** воспроизведения:

```sh
sudo docker exec -it tunnel-guard-live bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 topic echo /perception/status'
```

Терминал 3 — исходный PointCloud2 (его поступление, без вывода миллионов точек):

```sh
sudo docker exec -it tunnel-guard-live bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 topic hz /sensing/lidar/hesai128/pointcloud'
```

Терминал 4 — проигрывание bag. Для проверенной на сервере последовательности
используйте замедление `0.1`: это демонстрация функциональности, а не скорости.

```sh
sudo docker exec -it tunnel-guard-live bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag play /data/for_hackathon/doubleT_obstacle --rate 0.1'
```

Смотрите в `/perception/status` поля `status`, `objects`,
`nearest_obstacle_m` и `measurement_timestamp_ns`. Время
`callback_to_publish_s` выводится в журнале узла, не в сообщении статуса.
`nearest_obstacle_m` относится к ближайшему **подтверждённому опасному объекту**:
как с подтверждённым пересечением, так и с неразрешённым отношением к контуру.
Для подтверждённого пересечения отдельно используйте
`distance_summary.confirmed_intersection_m`. `/perception/attention_required` —
признак необходимости внимания,
**не команда торможения**. `no_obstacle_observed` также не доказывает
свободный путь. После остановки входа watchdog публикует `unknown` и очищает
живые маркеры. Перед повтором bag с начала перезапустите контейнер детектора:
временные метки должны возрастать монотонно.
После демонстрации остановите его командой `sudo docker stop tunnel-guard-live`.

## 4. Показать облако и результат в 3D

Для живой визуализации на машине с рабочим графическим сеансом и RViz2
откройте `rviz/tunnel_guard.rviz`. Конфигурация подписана на
`/perception/points_display` и `/perception/debug_markers`; фиксированная система
координат — `tunnel_guard_local`. Это координаты обработки, **не**
калиброванная система поезда. Облако в RViz разрежено только для показа:
детектор использует полные данные скана. Если на хосте есть ROS 2 Humble и
графический сеанс, из корня проекта запустите:

```sh
source /opt/ros/humble/setup.bash
rviz2 -d "$PWD/rviz/tunnel_guard.rviz"
```

На нынешнем сервере GUI RViz2 не проверялся.

Проверенный вариант для видеопоказа — браузерный 3D-просмотрщик **записанного
офлайн-прогона**. На текущем сервере уже запущены два локальных сервиса:
порт 8766 для `roundT_doubleT` и 8768 для `doubleT_obstacle`. На рабочей
машине с SSH-доступом оставьте туннель открытым:

```sh
ssh -N -L 127.0.0.1:8767:127.0.0.1:8766 \
  -L 127.0.0.1:8772:127.0.0.1:8768 USER@SERVER
```

Откройте `http://127.0.0.1:8767/3d` для полного проезда (252 кадра), затем
`http://127.0.0.1:8772/3d?frame=25&focus=160` для конкретного наблюдения
объекта на 55,8 м (второй полный прогон — 201 кадр). Эта страница **не
выполняет детектирование при воспроизведении**. Результаты и исходный bag
проверяются по идентичности источника и времени кадра.

Чтобы получить такой же записанный просмотр **на новом стенде**, выполните
отдельный полный офлайн-прогон. Рецепт
`configs/deployment-tz-demo.json` задаёт два bag, нативную конфигурацию,
`every: 1`, отсутствие ограничения кадров и каталог `/results/tz-demo-full`.
Каталог результата должен быть пустым: программа отказывается перезаписывать
существующие JSONL. Эта операция не используется вместо живой демонстрации
из раздела 3.

```sh
mkdir -p results
sudo docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$DEMO_DATA_DIR:/data:ro" -v "$PWD/results:/results" \
  -v "$PWD/configs/deployment-tz-demo.json:/recipes/demo.json:ro" \
  tunnel-guard:demo python3 -m tunnel_guard.run \
  --experiment /recipes/demo.json
```

После завершения откройте результат в отдельном контейнере. Порт 8768 должен
быть свободен; на текущем сервере этот просмотрщик уже запущен.

```sh
sudo docker run -d --name tunnel-guard-review-tz --restart unless-stopped \
  --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$DEMO_DATA_DIR:/data:ro" -v "$PWD/results:/results:ro" \
  tunnel-guard:demo python3 -m tunnel_guard.review_viewer \
  --run /results/tz-demo-full \
  --bag /data/for_hackathon/doubleT_obstacle \
  --port 8768 --display-max-points 60000
```

Откройте `http://127.0.0.1:8768/3d` на стенде или пробросьте порт через SSH,
как показано выше. После работы остановите контейнер
`sudo docker stop tunnel-guard-review-tz`. Существующие результаты
демонстрационного сервера описаны в
[SERVER_DEMO_20260927.md](SERVER_DEMO_20260927.md).

## 5. Параметры и приёмка

`input_topic`, `input_reliability`, `input_timeout_s` — параметры запуска
ROS 2-узла. `config` выбирает JSON-рецепт детектора; по умолчанию это
`/opt/tunnel-guard/configs/detector.json`. Основные физические
допущения в нём: преобразование системы сенсора, диапазон из `min_range_m`/`max_range_m`,
оценка грунта и рельсов, ширина колеи, опорный габарит, условия кластеризации
и временного подтверждения. Менять их под известный bag ради красивого
видео нельзя: на контрольных данных ТЗ это не будет обобщением. Состав
выходных тем и значения всех состояний приведены в [ROS2.md](ROS2.md).

Цепочка компонентов для сопроводительной схемы ТЗ:
`ROS 2 bag / PointCloud2 → декодирование и преобразование координат → оценка
пути и опорного габарита → кластеры и временные свидетельства → статус,
расстояние и маркеры`. Она показывает, какие именно данные проходят между
модулями; подробности алгоритма и его ограничения приведены в [README.md](../README.md).

Исторически на сервере с 16 GiB RAM и веткой `experiments/buyanov` проверено:
Docker-сборка; 10 исходных кадров через
живой ROS 2 при `0.1x` с совпадающими статусами офлайн-прогона; два полных
**офлайн**-прогона на 252 и 201 кадр; сохранённый 3D-просмотр. Их медианное
время обработки — 475 и 613 мс соответственно. При `0.25x` в ограниченном
живом прогоне были пропуски. Это не подтверждает работу в реальном времени,
точность на скрытом контрольном bag, предельную дальность обнаружения,
ложные тревоги по исчерпывающей разметке или безопасность движения.
Эти исторические замеры не квалифицируют текущую интеграционную ревизию.

В пакет сдачи по ТЗ нужно включить видео и явно указать, где находятся
описание архитектуры и алгоритма и отчёт об экспериментах. Полноценный прогон
на контрольном bag на **целевом** стенде ещё не выполнен.
В таком прогоне фиксируйте версию образа и конфигурации, фактическую тему и
QoS, число опубликованных результатов относительно входных кадров,
распределение задержек, использование CPU/GPU (`docker stats` во время
воспроизведения) и качество по независимой
разметке. Не называйте количество срабатываний на неразмеченных bag точностью
или частотой ложных тревог.
