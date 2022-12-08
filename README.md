# 6-DOF-Pose-Esimtation

Test set: https://drive.google.com/file/d/1isZRDidbJEjAuch678KNGgI-yV6z00Ll/view?usp=sharing

How to run:

- Trained u-net model is provided in this link: https://github.com/SrinidhiBharadwaj/6-DOF-Pose-Esimtation/blob/main/saved_models_unet_a_new_hope/best_model.pt (stored segmentation masks is provided in the drive)
- Download this model and use it in evaluation.py to generate segmentation masks
- This saved the segmentation images in the same folder with the name "image-name_unet.png"
- Run "pose_estimation.py" script to generate the poses
  - This script also generates bounding boxes in "bboxes" folder.
  - "get_pose_new" method in this script is the core function that estimates poses.
