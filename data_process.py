import pickle
import pandas

def main():
    file_path = "./dataset/yue_zh_combined36k.pkl"
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # 定义输出文件路径
    output_file = './dataset/output.txt'

    # 将DataFrame转换为txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历DataFrame的每一行
        for index, row in data.iterrows():
            # 获取input_text和target_text的值
            input_text = row['input_text']
            target_text = row['target_text']

            # 按指定格式拼接字符串
            line = f"{input_text}:{target_text}\n"

            # 将拼接后的字符串写入txt文件
            f.write(line)

if __name__ == "__main__":
    main()
