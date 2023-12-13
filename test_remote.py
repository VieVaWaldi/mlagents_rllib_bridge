import ray
import logging
import socket
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO)

@ray.remote(resources={"worker_node": 1})
def test_remote_function():
    hostname = socket.gethostname()
    logging.info(f"Running on hostname: {hostname}")

    # Log some environment variables or other details
    env_vars = os.environ
    for key, value in env_vars.items():
        logging.info(f"ENV {key}: {value}")

    return ray.worker.global_worker.node.unique_id

if __name__ == "__main__":
    ray.init(address='192.168.193.41:6379')
    result_id = test_remote_function.remote()
    result = ray.get(result_id)
    print("Result (Node ID):", result)
    ray.shutdown()






# import ray

# # Initialize Ray
# ray.init(address='192.168.193.41:6379')  # Use the address of your Ray head node

# # Define a simple remote function
# @ray.remote(resources={"worker_node": 1})
# def test_remote_function():
#     return ray.worker.global_worker.node.unique_id  # Return the unique ID of the executing node

# if __name__ == "__main__":
#     # Get the list of all nodes in the cluster

#     print("{#} MAIN")

#     all_nodes = ray.nodes()
#     # print(all_nodes)

#     target_node = all_nodes[0]
#     print("{#} TARGET NODE")
#     print(target_node)
#     print("{#} -----------")

#     # Run the remote function on the selected node
#     result_id = test_remote_function.remote()

#     # Get the result from the selected node
#     result = ray.get(result_id)

#     print("Result (Node ID):", result)

#     # Shutdown Ray
#     ray.shutdown()
