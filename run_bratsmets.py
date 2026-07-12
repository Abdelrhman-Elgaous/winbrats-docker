"""BraTS-METS 2026 entrypoint: /input -> /output, weights baked at /opt/weights."""
import infer_logic
if __name__ == "__main__":
    infer_logic.run_inference(data_path="/input", output_path="/output", weights="/opt/weights")
