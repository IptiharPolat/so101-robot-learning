# Evaluation rubric

## Trial fields

`first_object_correct`, `first_grasp_success`, `first_place_success`, `first_release_before_second`, `second_object_correct`, `second_grasp_success`, `second_place_success`, `both_inside_center`, `order_success`, `end_to_end_success`, `collision`, `timeout`, and `failure_stage`.

For SmolVLA also record the selected colors, `non_target_contact_count`, and whether switching the instruction on the same layout switched the first target.

## Failure taxonomy

- `wrong_first_object`
- `wrong_second_object`
- `order_violation`
- `first_grasp_failure`
- `first_place_failure`
- `first_release_failure`
- `second_grasp_failure`
- `second_place_failure`
- `object_collision`
- `center_area_collision`
- `non_target_contact`
- `object_slip`
- `camera_occlusion`
- `workspace_limit`
- `policy_oscillation`
- `timeout`
- `software_or_hardware_error`

`end_to_end_success` requires correct first object, successful first grasp/place/release, correct second object, successful second grasp/place, both cubes inside the target, correct order, and no disqualifying collision/timeout.
