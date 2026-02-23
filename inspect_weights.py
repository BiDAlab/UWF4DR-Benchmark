import h5py
import sys

def inspect_h5(path):
    print(f"\n=== Inspecting: {path} ===\n")
    with h5py.File(path, "r") as f:

        all_weight_names = []
        def visit(name, obj):
            if hasattr(obj, "attrs") and "weight_names" in obj.attrs:
                wn = obj.attrs["weight_names"]
                for w in wn:
                    all_weight_names.append(
                        w.decode("utf-8") if isinstance(w, bytes) else str(w)
                    )
        f.visititems(visit)

        print("TOTAL WEIGHT NAMES:", len(all_weight_names))
        print("\nFirst 40 weights:")
        for w in all_weight_names[:40]:
            print(" ", w)

        print("\n=== CLASSIFIER-RELATED WEIGHTS ===")
        for w in all_weight_names:
            low = w.lower()
            if any(k in low for k in ["head", "classifier", "dense", "logits", "pred"]):
                print(" ", w)

        print("\n=== TOP-LEVEL GROUPS ===")
        print(list(f.keys()))

if __name__ == "__main__":
    inspect_h5(sys.argv[1])
