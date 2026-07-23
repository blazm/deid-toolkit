# TODO / Upcoming

## Pose estimation before vs. after de-identification
Estimate head pose (yaw, pitch, roll) on original and de-identified images to verify that pose data utility is preserved. This is critical for downstream tasks like pose-invariant recognition and surveillance analytics.

## Gaze estimation before vs. after de-identification
Estimate gaze direction on original and de-identified images to verify that gaze data utility is preserved. Gaze is a sensitive attribute that can leak intent; verifying its preservation helps assess whether de-identification inadvertently compromises or leaks behavioral signals.

## FIQ (Face Image Quality) metric before vs. after de-identification
Compute FIQ scores on original and de-identified images to evaluate whether de-identification degrades image quality below usable thresholds. FIQ assesses multiple quality dimensions (e.g., illumination, contrast, resolution, facial region quality) and provides an overall quality score. This helps determine if de-identified faces still meet quality requirements for downstream face recognition systems.
