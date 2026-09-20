# Sensor specification evidence — 2026-09-16

User supplied [Hesai downloads / Pandar128](https://www.hesaitech.com/downloads/#pandar128).
The linked manual is [Pandar128E3X v4p5, 128-en-251110](https://www.hesaitech.com/wp-content/uploads/2025/11/Pandar128E3X_v4p5_User_Manual_128-en-251110.pdf).
The same manual is now preserved locally as `ALL WE KNOW /описание лидара.pdf`.
This identifies a lidar family, not the exact device/firmware/driver in our bags.

Manufacturer statements (PDF page numbers, one-based):

- Pages 15–18: rotation axis Z, azimuth zero along Y, nonuniform vertical channel angles;
  accurate angular corrections are device-specific.
- Page 19: 10/20 Hz; single and dual return modes; range specifications depend on reflectivity.
- Page 19: the stated 0.3–200 m capability is specified at 10% reflectivity; it is not an
  obstacle-detection guarantee.
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

The Q&A transcript in `ALL WE KNOW /All We Know.md` adds an informal mount-height report:
about 1.075 m on recordings without staged obstacles and about 1.5 m on another carrier used
for the obstacle recording. This is useful evidence that the mount is not fixed, but it is not a
survey. The detector therefore continues to fit the local bed rather than switching on a bag name.
It now reports `estimated_sensor_height_from_local_bed_m`; first-frame estimates across the six
recordings are 1.305–1.720 m and remain explicitly marked as local geometric estimates.

No timing, deskew, sensor transform or detector threshold was changed based on the manual.
