import tensorflow as tf

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Replace with the path to your checkpoint
checkpoint_path = "/home/ytj/carla-ros-bridge/src/carla_ros2_lane_keep/carla_ros2_lane_keep/models/LaneNet/weights/tusimple_lanenet.ckpt"
new_checkpoint_path = "/home/ytj/carla-ros-bridge/src/carla_ros2_lane_keep/carla_ros2_lane_keep/models/LaneNet/weights/new/tusimple_lanenet.ckpt"

# Load the original checkpoint
reader = tf.train.load_checkpoint(checkpoint_path)
variables = tf.train.list_variables(checkpoint_path)

# Define multiple prefix mappings
substr_mappings = {
    "/bn/":
    "/cond/bn/",
    "/pix_bn/":
    "/cond/pix_bn/",
    "/bn_1/":
    "/cond/bn_1/",
    "/dw_bn/":
    "/cond/dw_bn/",
    "/dw_bn_1/":
    "/cond_1/dw_bn_1/",
    "/dw_bn_2/":
    "/cond_2/dw_bn_2/",
    "/input_project_bn/":
    "/cond/input_project_bn/",
}

# Create a dictionary to hold renamed variables
renamed_variables = {}

# Rename variables based on prefix mappings
for var_name, shape in variables:
    new_name = var_name  # Default to original name
    for old_substr, new_substr in substr_mappings.items():
        if old_substr in var_name:
            new_name = var_name.replace(old_substr, new_substr)
            break  # Exit loop once a match is found
    renamed_variables[new_name] = tf.compat.v1.get_variable(new_name, initializer=reader.get_tensor(var_name))

# Save the renamed variables into a new checkpoint
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(var_list=renamed_variables)
    saver.save(sess, new_checkpoint_path)

print("New checkpoint saved with renamed variables.")

# new_variables = tf.train.list_variables(new_checkpoint_path)
#
# print("Variables in the new checkpoint:")
# for var_name, shape in new_variables:
#     print(f"{var_name}: {shape}")
