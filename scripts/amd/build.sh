set -x

cd $AMD_PROJECT_HOME/python
pip uninstall -y triton

bash $AMD_PROJECT_HOME/scripts/amd/clean.sh

export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1
export AMDGCN_ENABLE_DUMP=1

export TRITON_USE_ROCM=ON

# pip install -U matplotlib pandas filelock tabulate
pip install --verbose -e .
