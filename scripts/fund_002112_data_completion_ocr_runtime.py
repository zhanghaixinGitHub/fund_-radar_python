"""限制本轮 OCR 的线程，避免多个扫描任务占满本机处理器。

只改变本进程创建的推理会话，模型文件及识别参数不变；逐页证据可恢复。
"""
def engine():
    import rapidocr_onnxruntime.utils as runtime
    from onnxruntime import SessionOptions
    from rapidocr_onnxruntime import RapidOCR

    def options():
        value = SessionOptions()
        value.intra_op_num_threads = 2
        value.inter_op_num_threads = 1
        value.add_session_config_entry('session.intra_op.allow_spinning', '0')
        value.add_session_config_entry('session.inter_op.allow_spinning', '0')
        return value

    runtime.SessionOptions = options
    return RapidOCR()
