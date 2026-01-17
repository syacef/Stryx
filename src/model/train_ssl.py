from model.ssl.pipeline import SSLPipeline, StudentType, TeacherType

if __name__ == "__main__":
    RAW_DATA_PATH = "./data"
    TEACHER_BACKBONE = TeacherType.DINOV2_VITS14.value
    STUDENT_BACKBONE = StudentType.TINYVIT.value

    pipeline = SSLPipeline(
        teacher_type=TEACHER_BACKBONE,
        student_type=STUDENT_BACKBONE,
        base_data_dir=RAW_DATA_PATH,
    )
    pipeline.extract_features(RAW_DATA_PATH)
    pipeline.train_ssl(
        video_input_dir=RAW_DATA_PATH, epochs=30, batch_size=32, lr=1e-4, start_epoch=22
    )
