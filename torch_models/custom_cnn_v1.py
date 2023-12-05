import torch
import torch.nn as nn
from gym import Space

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork


class custom_cnn_v1(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # CNN layers for 2-channel image observations
        self.cnn_layers_2ch = nn.Sequential(
            nn.Conv2d(2, 32, [3, 3], stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, [3, 3], stride=2),
            nn.ReLU(),
            # More layers or flattening as needed
        )

        # CNN layers for 6-channel image observation
        self.cnn_layers_6ch = nn.Sequential(
            nn.Conv2d(6, 32, [3, 3], stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, [3, 3], stride=2),
            nn.ReLU(),
            # More layers or flattening as needed
        )

        self.fc_for_vector = FullyConnectedNetwork(
            Space(shape=(31,)), action_space, num_outputs, model_config, name
        )

        cnn_output_size = (4 * 13824)  # Printed from out shape
        vector_output_size = 10  # torch.Size([32, 10])
        num_combined_features = cnn_output_size + vector_output_size  # torch.Size([32, 55306])

        # Adjust the final fully connected layers
        self.final_fc = FullyConnectedNetwork(Space(shape=(num_combined_features,)), action_space, num_outputs,
                                              model_config, name)

    def forward(self, input_dict, state, seq_lens):
        # Assuming the first four elements of the tuple are image observations
        # and the last element is the vector observation.
        print("HEEEERE")
        print(input_dict)
        print(dir(input_dict))

        image_obs_list = input_dict["obs"][:4]  # List of image observations
        vector_obs = input_dict["obs"][4]  # Vector observation

        # for i, img_obs in enumerate(image_obs_list):
        #     print(f"Image Observation {i} - {img_obs.shape}")
        #
        # # Print min and max for the vector observation
        # print(f"Vector Observation - {vector_obs.shape}")

        # Process each image observation through the CNN and combine
        cnn_outputs = []
        for i, img_obs in enumerate(image_obs_list):
            if i != 2:  # For 2-channel images
                cnn_out = self.cnn_layers_2ch(img_obs)
            else:  # For the 6-channel image
                cnn_out = self.cnn_layers_6ch(img_obs)
            cnn_out_flat = torch.flatten(cnn_out, start_dim=1)
            # print(cnn_out_flat.shape)
            cnn_outputs.append(cnn_out_flat)

        # Concatenate all CNN outputs
        combined_cnn_out = torch.cat(cnn_outputs, dim=1)

        # Process vector observation through fully connected layers
        vector_output_tuple = self.fc_for_vector({"obs": vector_obs})
        # Extract the tensor from the tuple
        vector_out = vector_output_tuple[0] if isinstance(vector_output_tuple, tuple) else vector_output_tuple

        # Combine CNN and vector outputs
        combined_out = torch.cat([combined_cnn_out, vector_out], dim=1)
        # print(combined_out.shape)

        # Final fully connected layers
        output = self.final_fc({"obs": combined_out})
        # print(output)
        return output[0], state

    def value_function(self):
        return self.final_fc.value_function()
