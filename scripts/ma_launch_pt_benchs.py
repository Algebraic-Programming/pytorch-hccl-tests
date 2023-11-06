"""
ModelArts entrypoint for PyTorch HCCL benchmarks
"""
import argparse
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


# Input argparams
parser = argparse.ArgumentParser(
    description="Modelarts PyTorch HCCL benchmarks entrypoint"
)
parser.add_argument(
    "--benchmark", type=str, default=os.environ.get("PT_HCCL_BENCHMARK", "allreduce")
)
parser.add_argument(
    "--code-dir",
    type=str,
    default=os.environ.get("PT_HCCL_CODE_DIR", "pytorch-hccl-benchs"),
)
parser.add_argument("--nnodes", type=int, default=os.environ.get("MA_NUM_HOSTS", 1))
parser.add_argument("--nproc_per_node", type=int, default=8)

args, unparsed = parser.parse_known_args()


def main():
    logger.info("*" * 50)
    logger.info(f"* Hostname                              : {os.system('hostname')}")
    logger.info(f"* HCCL benchmark                        : {args.benchmark}")
    logger.info(f"* Number of nodes (--nnodes)            : {args.nnodes}")
    logger.info(f"* Processes per node: (--nproc_per_node): {args.nproc_per_node}")
    logger.info("*" * 50)

    print("os.environ: ", os.environ)

    benchmark = args.benchmark
    hccl_bench_code_dir = args.code_dir

    # get environment variables set by ModelArts job
    host = "{0}-{1}-0.{2}".format(
        os.environ["MA_VJ_NAME"], os.environ["MA_TASK_NAME"], os.environ["MA_VJ_NAME"]
    )
    port = "6789"
    rdzv_endpoint = f"{host}:{port}"

    node_rank = (
        int(os.environ["VC_TASK_INDEX"])
        if os.environ.get("VC_TASK_INDEX") is not None
        else int(os.environ.get("MA_TASK_INDEX"))
    )

    logger.info("*" * 50)
    logger.info(f"* host: {host}; port: {port}")
    logger.info(f"* rdzv_endpoint: {rdzv_endpoint}")
    logger.info(f"* Node_rank: {node_rank}")
    logger.info(f"* MA_JOB_DIR: {node_rank}")
    logger.info("*" * 50)

    # Torchrun distrbuted job
    main_file_local = os.path.join(
        os.environ["MA_JOB_DIR"],
        hccl_bench_code_dir,
        "pytorch_hccl_tests/cli.py",
    )
    cmd = (
        f"torchrun --nproc_per_node={args.nproc_per_node} --nnodes {args.nnodes} --node_rank={node_rank}"
        f" --master_addr {host} --master_port {port} {main_file_local} --benchmark {benchmark} --device npu"
    )

    logger.info(f"cmd: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
