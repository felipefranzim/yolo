import os

def check_labels(path):
    ok = 0
    fail = 0
    for fname in os.listdir(path):
        if fname.endswith('.txt'):
            with open(os.path.join(path, fname)) as f:
                try:
                    for line in f:
                        parts = line.strip().split()
                        assert len(parts) == 5
                        cls = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        assert all(0 <= v <= 1 for v in coords)
                    ok += 1
                except Exception as e:
                    print(f"Erro no arquivo {fname}: {e}")
                    fail += 1
    print(f"Labels OK: {ok}, Labels com erro: {fail}")

print('Validação train:')
check_labels('dataset/labels/train')
print('Validação val:')
check_labels('dataset/labels/val')