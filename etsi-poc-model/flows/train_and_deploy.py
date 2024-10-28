from prefect import Flow
from prefect.tasks.prefect import create_flow_run, wait_for_flow_run

with Flow("Parent Flow") as flow:

    # assumes you have registered the following flows in a project named "examples"
    flow_a = create_flow_run(flow_name="MLbasic")
    wait_for_flow_a = wait_for_flow_run(flow_a, raise_final_state=True)

    flow_b = create_flow_run(flow_name="deploy")
    wait_for_flow_b = wait_for_flow_run(flow_b, raise_final_state=True)

    flow_b.set_upstream(wait_for_flow_a)

if __name__ == "__main__":
    flow.run()
