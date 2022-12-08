from mymodel import Inference


def main():
    text = input("Nhập văn bản: ")
    print("*"*10)
    print("checkpoint được lưu theo epoch")
    print("*"*10)
    checkpoint_ind = input("Nhập số thứ tự của checkpoint: ")
    config_dir = '/home/vnpt/config_dir/config.json'
    inference = Inference(config_dir=config_dir, checkpoint_ind=checkpoint_ind)
    inference.infer(text=text)


if __name__ == '__main__':
    main()

