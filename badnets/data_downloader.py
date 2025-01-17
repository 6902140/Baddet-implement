import pathlib
from dataset import build_init_data

# 下載 mnist和cifar-10數據集
def main():
    data_path = './data/'
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True) 
    build_init_data('MNIST',True, data_path)
    build_init_data('CIFAR10',True, data_path)

if __name__ == "__main__":
    main()
