from ultralytics import YOLO
import time

def train_yolo():
    print("This is train!!!!!!!!!!!!!!!!")
    # 加载 YOLOv8 预训练模型
    model = YOLO("yolov8s.pt")

    # 训练参数
    model.train(
        data="./datasets/data.yaml",  
        device="cuda",  # 使用 GPU
        project="runs/refine_train",
        name="refine_train",
        batch = 64,
        lr0=0.0006527556335135585,
        optimizer = "Adam",
        epochs=5
        # 尝试调整数据集增强
        #mosaic=1.0,
        #imgsz=768,
        #scale=0.5,
        
    )
def export_val_predictions():
    print("This is val!!!!!!!!!!!!!!!!!!!!")
    model = YOLO(r"./runs/refine_train/weights/best.pt")

    # 预测验证集（val）上的所有图像并保存结果
    results = model.predict(
        source=r"./datasets/valid/images",  # val 图像路径
        conf=0.25,                # 置信度阈值
        save=True,                # 保存预测图像
        save_txt=True,            # 保存预测框标签文件（YOLO格式）
        save_conf=True,           # 保存每个框的置信度
        project="runs/val_predict",   # 预测保存路径
        name="best_val_output",       # 子目录名称
    )
    print("✅ 所有验证图像的预测结果已保存！")  


def export_test_predictions():
    print("This is val!!!!!!!!!!!!!!!!!!!!")
    model = YOLO(r"./runs/refine_train/weights/best.pt")

    # 预测验证集（val）上的所有图像并保存结果
    results = model.predict(
        source=r"./datasets/test/images",  # val 图像路径
        conf=0.25,                # 置信度阈值
        save=True,                # 保存预测图像
        save_txt=True,            # 保存预测框标签文件（YOLO格式）
        save_conf=True,           # 保存每个框的置信度
        project="runs/test_predict",   # 预测保存路径
        name="best_test_output",       # 子目录名称
    )
    print("✅ 所有验证图像的预测结果已保存！")  
def test_yolo():
    # 加载训练完成的模型
    print("This is test!!!!!!!!!!!!!!!!")
    model = YOLO(r"./runs/refine_train/weights/best.pt")
    # 在测试集上评估模型
    metrics = model.val(data="./datasets/data.yaml", split="test")
    
    print(metrics) 
if __name__ == "__main__":
    train_yolo()
    export_val_predictions()
    export_test_predictions()
    test_yolo()