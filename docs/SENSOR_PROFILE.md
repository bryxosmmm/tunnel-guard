# Sensor specification evidence — 2026-09-20

User supplied [Hesai downloads / Pandar128](https://www.hesaitech.com/downloads/#pandar128).
The linked manual is [Pandar128E3X v4p5, 128-en-251110](https://www.hesaitech.com/wp-content/uploads/2025/11/Pandar128E3X_v4p5_User_Manual_128-en-251110.pdf).
This identifies a lidar family, not the exact device/firmware/driver in our bags.

The local manual supplied for this iteration is `Pandar128E3X_v4p5_User_Manual_128-en-240710-1.pdf` (150 pages, document version `128-en-240710`). It is consistent with the family manual and adds useful packet-level evidence:

- Pages 15–16: 128 channels; 360° horizontal FOV; vertical FOV −25°…+15°; 10/20 Hz; 0.1°/0.2° at 10 Hz or 0.2°/0.4° at 20 Hz; nominal ranging capability 0.3–200 m at 10% reflectivity; single or dual return; GPS/PTP clock source.
- Pages 40–48: a point-cloud UDP packet contains two 128-channel blocks, 4 mm distance units, reflectivity, return mode, motor speed, UTC date/time, microsecond timestamp, UDP sequence and optional IMU fields. Dual-return blocks can repeat the same physical return.
- Pages 50 and 116–124: per-point timing requires packet time, block timing, operational/azimuth state and the channel firing-offset table; the accurate horizontal/vertical angles come from the shipped angle-correction file.
- Pages 68–70: noise, interstitial, retro-multi-reflection and up-close blockage filtering are configurable and default to off in the documented interface; the actual device settings are still unknown.
- Pages 113–115: GPS can lose lock and drift; PTP can be free-running, tracking, locked or frozen. A ROS header timestamp alone does not prove the active clock source.

Manufacturer statements (PDF page numbers, one-based):

- Pages 15–18: rotation axis Z, azimuth zero along Y, nonuniform vertical channel angles;
  accurate angular corrections are device-specific.
- Page 19: 10/20 Hz; single and dual return modes; range specifications depend on reflectivity.
- Page 51: dual returns occupy adjacent blocks from the same firing; some modes may
  repeat identical returns. They are not independent temporal confirmations.
- Pages 119–120: point firing times require packet/block timing plus channel offsets.

Project implications:

- Do not flip `sensor_profile_verified` merely from this manual. Obtain exact model,
  calibration file, installed transform and driver configuration/version.
- Sensor housing axes do not establish how PointCloud2 was transformed by its producer.
  Current x=-raw_y, y=raw_x, z=raw_z is still an unverified installation assumption.
- Do not infer return mode from PointCloud2 array capacity or ring count alone.
- Packet timestamps described here do not prove the meaning of our FLOAT64 `timestamp`
  field. Confirm driver conversion and whether deskew/aggregation was already performed.
- Ask for scan assembly mode, return mode, firing/angle correction files, sensor-to-vehicle
  transform, hardware filtering settings and clock source. Q&A may resolve these questions.
- No advertised hardware range is a measured obstacle-detection range on these recordings.
- The detector's current `max_range_m=220` is a software acceptance bound inherited from the
  research recipe, not a Pandar128E3X capability claim. Reconcile it with the exact channel
  table, reflectivity and driver filtering before using range bins as an acceptance result.

Additional questions made concrete by this manual:

- Was the PointCloud2 produced by the Hesai driver from raw packets, or by a recorder after
  packet aggregation? If from the driver, which return mode, spin rate, trigger method and
  filtering settings were active?
- Was the supplied per-device angle-correction file loaded by the driver? The bag's `ring`
  field is not a substitute for the per-channel horizontal offsets or firing times.
- Does the FLOAT64 `timestamp` field contain each laser firing time, a block time, or a
  driver-normalized value? The manual's packet timestamp is UTC microseconds, while the ROS
  message header is a separate acquisition clock.
- Were GPS or PTP connected and locked during the recording? If PTP was frozen or GPS lost,
  the absolute packet time can drift even though packets continue to arrive.
- Is the device in single or dual return mode? Two adjacent blocks in dual mode must not be
  counted as two independent temporal observations.

No timing, deskew, sensor transform or detector threshold was changed based on the manual.
