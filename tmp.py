GILM_Clip(
  (GILM_Clip): VSSM(
    (backbone): CLIP(
      (visual): VSSM(
        (patch_embed): Sequential(
          (0): Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (1): Permute()
          (2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          (3): Permute()
          (4): GELU(approximate='none')
          (5): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (6): Permute()
          (7): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
        (layers): ModuleList(
          (0): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=96, out_features=192, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
                  (out_proj): Linear(in_features=192, out_features=96, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.0)
                (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=96, out_features=384, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=384, out_features=96, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=96, out_features=192, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
                  (out_proj): Linear(in_features=192, out_features=96, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.015000000596046448)
                (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=96, out_features=384, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=384, out_features=96, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Sequential(
              (0): Permute()
              (1): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (2): Permute()
              (3): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
          )
          (1): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=192, out_features=384, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
                  (out_proj): Linear(in_features=384, out_features=192, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.030000001192092896)
                (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=192, out_features=768, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=768, out_features=192, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=192, out_features=384, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
                  (out_proj): Linear(in_features=384, out_features=192, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.04500000178813934)
                (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=192, out_features=768, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=768, out_features=192, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Sequential(
              (0): Permute()
              (1): Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (2): Permute()
              (3): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            )
          )
          (2): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.06000000238418579)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.07500000298023224)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (2): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.09000000357627869)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (3): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.10500000417232513)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (4): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.12000000476837158)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (5): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.13500000536441803)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (6): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.15000000596046448)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (7): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.16500000655651093)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (8): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.18000000715255737)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (9): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.19500000774860382)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (10): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.21000000834465027)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (11): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.22500000894069672)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (12): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.24000000953674316)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (13): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.2549999952316284)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (14): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.27000001072883606)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Sequential(
              (0): Permute()
              (1): Conv2d(384, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (2): Permute()
              (3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            )
          )
          (3): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=768, out_features=1536, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
                  (out_proj): Linear(in_features=1536, out_features=768, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.2850000262260437)
                (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=768, out_features=3072, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=3072, out_features=768, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=768, out_features=1536, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
                  (out_proj): Linear(in_features=1536, out_features=768, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.30000001192092896)
                (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=768, out_features=3072, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=3072, out_features=768, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Identity()
          )
        )
        (classifier): Sequential(
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (permute): Permute()
          (avgpool): AdaptiveAvgPool2d(output_size=1)
          (flatten): Flatten(start_dim=1, end_dim=-1)
          (head): Identity()
        )
      )
      (visual_e): VSSM(
        (patch_embed): Sequential(
          (0): Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (1): Permute()
          (2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          (3): Permute()
          (4): GELU(approximate='none')
          (5): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (6): Permute()
          (7): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
        (layers): ModuleList(
          (0): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=96, out_features=192, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
                  (out_proj): Linear(in_features=192, out_features=96, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.0)
                (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=96, out_features=384, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=384, out_features=96, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=96, out_features=192, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
                  (out_proj): Linear(in_features=192, out_features=96, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.015000000596046448)
                (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=96, out_features=384, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=384, out_features=96, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Sequential(
              (0): Permute()
              (1): Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (2): Permute()
              (3): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
          )
          (1): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=192, out_features=384, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
                  (out_proj): Linear(in_features=384, out_features=192, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.030000001192092896)
                (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=192, out_features=768, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=768, out_features=192, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=192, out_features=384, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
                  (out_proj): Linear(in_features=384, out_features=192, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.04500000178813934)
                (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=192, out_features=768, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=768, out_features=192, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Sequential(
              (0): Permute()
              (1): Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (2): Permute()
              (3): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            )
          )
          (2): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.06000000238418579)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.07500000298023224)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (2): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.09000000357627869)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (3): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.10500000417232513)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (4): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.12000000476837158)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (5): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.13500000536441803)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (6): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.15000000596046448)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (7): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.16500000655651093)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (8): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.18000000715255737)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (9): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.19500000774860382)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (10): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.21000000834465027)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (11): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.22500000894069672)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (12): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.24000000953674316)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (13): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.2549999952316284)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (14): VSSBlock(
                (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=384, out_features=768, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
                  (out_proj): Linear(in_features=768, out_features=384, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.27000001072883606)
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=384, out_features=1536, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=1536, out_features=384, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Sequential(
              (0): Permute()
              (1): Conv2d(384, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              (2): Permute()
              (3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            )
          )
          (3): Sequential(
            (blocks): Sequential(
              (0): VSSBlock(
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=768, out_features=1536, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
                  (out_proj): Linear(in_features=1536, out_features=768, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.2850000262260437)
                (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=768, out_features=3072, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=3072, out_features=768, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (1): VSSBlock(
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (op): SS2D(
                  (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
                  (in_proj): Linear(in_features=768, out_features=1536, bias=False)
                  (act): SiLU()
                  (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
                  (out_proj): Linear(in_features=1536, out_features=768, bias=False)
                  (dropout): Identity()
                )
                (drop_path): timm.DropPath(0.30000001192092896)
                (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=768, out_features=3072, bias=True)
                  (act): GELU(approximate='none')
                  (fc2): Linear(in_features=3072, out_features=768, bias=True)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
            )
            (downsample): Identity()
          )
        )
        (classifier): Sequential(
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (permute): Permute()
          (avgpool): AdaptiveAvgPool2d(output_size=1)
          (flatten): Flatten(start_dim=1, end_dim=-1)
          (head): Identity()
        )
      )
      (transformer): Transformer(
        (resblocks): Sequential(
          (0): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (1): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (2): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (3): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (4): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (5): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (6): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (7): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (8): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (9): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (10): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (11): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (transformer_e): Transformer(
        (resblocks): Sequential(
          (0): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (1): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (2): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (3): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (4): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (5): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (6): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (7): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (8): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (9): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (10): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (11): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=512, out_features=2048, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=2048, out_features=512, bias=True)
            )
            (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (token_embedding): Embedding(49408, 512)
      (ln_final): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (token_embedding_e): Embedding(49408, 512)
      (ln_final_e): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (normal_mlp): NormalMLP(
      (linear): Linear(in_features=512, out_features=512, bias=True)
      (activation): ReLU()
      (dropout): Dropout(p=0.3, inplace=False)
      (projection): Linear(in_features=512, out_features=512, bias=False)
      (fc): Linear(in_features=512, out_features=1, bias=False)
    )
    (clslayer): Classifier_Clip(
      (classifier): Sequential(
        (0): Linear(in_features=512, out_features=1, bias=True)
      )
    )
    (patch_embed): PatchEmbed2D(
      (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.0)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.02857142873108387)
          )
        )
        (downsample): PatchMerging2D(
          (reduction): Linear(in_features=384, out_features=192, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.05714285746216774)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.08571428805589676)
          )
        )
        (downsample): PatchMerging2D(
          (reduction): Linear(in_features=768, out_features=384, bias=False)
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.11428571492433548)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.1428571492433548)
          )
        )
        (downsample): PatchMerging2D(
          (reduction): Linear(in_features=1536, out_features=768, bias=False)
          (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.17142857611179352)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.20000000298023224)
          )
        )
      )
    )
    (layers_up): ModuleList(
      (0): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.20000000298023224)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.1666666716337204)
          )
        )
      )
      (1): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.13333332538604736)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.09999999403953552)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=768, out_features=1536, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.06666667014360428)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.03333333507180214)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=384, out_features=768, bias=False)
          (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.0)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=192, out_features=384, bias=False)
          (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (PRIM4): PRIM4(
      (MOA): MOA(
        (layers): Sequential(
          (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): dilatedComConv4(
            (conv2_1): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (conv2_2): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
            (conv2_3): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=(3, 3), groups=8, bias=False)
            (conv2_4): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
          )
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
          (6): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): ReLU(inplace=True)
        )
      )
      (MSA): MSA4(
        (reductionLayers): Sequential(
          (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (get_overlap_patches): Unfold(kernel_size=3, dilation=1, padding=1, stride=2)
        (overlap_embed): Conv2d(4608, 512, kernel_size=(1, 1), stride=(1, 1))
        (SelfAttention): EfficientSelfAttention(
          (to_qkv): Conv2d(512, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (to_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (ffd): MixFeedForward(
          (net): Sequential(
            (0): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): GELU()
            (3): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (LN): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (merge_context): MergePR(
        (features): Sequential(
          (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
    )
    (PRIM3): PRIM3(
      (MOA): MOA(
        (layers): Sequential(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): dilatedComConv4(
            (conv2_1): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (conv2_2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
            (conv2_3): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=(3, 3), groups=8, bias=False)
            (conv2_4): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
          )
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
          (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): ReLU(inplace=True)
        )
      )
      (MSA): MSA3(
        (reductionLayers): Sequential(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (get_overlap_patches): Unfold(kernel_size=3, dilation=1, padding=1, stride=2)
        (overlap_embed): Conv2d(2304, 256, kernel_size=(1, 1), stride=(1, 1))
        (SelfAttention): EfficientSelfAttention(
          (to_qkv): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (to_out): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (ffd): MixFeedForward(
          (net): Sequential(
            (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): GELU()
            (3): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (LN): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (merge_context): MergePR(
        (features): Sequential(
          (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
    )
    (PRIM2): PRIM2(
      (MOA): MOA(
        (layers): Sequential(
          (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): dilatedComConv4(
            (conv2_1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (conv2_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
            (conv2_3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=(3, 3), groups=8, bias=False)
            (conv2_4): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
          )
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
          (6): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): ReLU(inplace=True)
        )
      )
      (MSA): MSA2(
        (reductionLayers): Sequential(
          (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (get_overlap_patches): Unfold(kernel_size=3, dilation=1, padding=1, stride=2)
        (overlap_embed): Conv2d(1152, 128, kernel_size=(1, 1), stride=(1, 1))
        (SelfAttention): EfficientSelfAttention(
          (to_qkv): Conv2d(128, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (to_out): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (ffd): MixFeedForward(
          (net): Sequential(
            (0): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): GELU()
            (3): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (LN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (merge_context): MergePR(
        (features): Sequential(
          (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
    )
    (PRIM1): PRIM1(
      (MOA): MOA(
        (layers): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): dilatedComConv4(
            (conv2_1): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (conv2_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
            (conv2_3): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=(3, 3), groups=8, bias=False)
            (conv2_4): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
          )
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
          (6): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): ReLU(inplace=True)
        )
      )
      (MSA): MSA1(
        (reductionLayers): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (get_overlap_patches): Unfold(kernel_size=3, dilation=1, padding=1, stride=2)
        (overlap_embed): Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
        (SelfAttention): EfficientSelfAttention(
          (to_qkv): Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (to_out): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (ffd): MixFeedForward(
          (net): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): GELU()
            (3): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (LN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
      (merge_context): MergePR(
        (features): Sequential(
          (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
    )
    (PRM3): PRM3(
      (above_conv2): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (above_bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (right_conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (right_bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fuse_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fuse_bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (CAM): CAM(
        (softmax): Softmax(dim=-1)
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      )
      (FAPA): FAPA(
        (softmax): Softmax(dim=-1)
        (FAPAEnc): FAPAEnc(
          (pool1): DAP(
            (conv1x1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool2): DAP(
            (conv1x1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool3): DAP(
            (conv1x1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool4): DAP(
            (conv1x1): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (conv1): Sequential(
            (0): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv2): Sequential(
            (0): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv3): Sequential(
            (0): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv4): Sequential(
            (0): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (conv_query): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (conv_key): Linear(in_features=512, out_features=128, bias=True)
        (conv_value): Linear(in_features=512, out_features=512, bias=True)
      )
    )
    (PRM2): PRM2(
      (above_conv2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (above_bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (right_conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (right_bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fuse_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fuse_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (CAM): CAM(
        (softmax): Softmax(dim=-1)
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (FAPA): FAPA(
        (softmax): Softmax(dim=-1)
        (FAPAEnc): FAPAEnc(
          (pool1): DAP(
            (conv1x1): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool2): DAP(
            (conv1x1): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool3): DAP(
            (conv1x1): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool4): DAP(
            (conv1x1): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv3): Sequential(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv4): Sequential(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (conv_query): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (conv_key): Linear(in_features=256, out_features=64, bias=True)
        (conv_value): Linear(in_features=256, out_features=256, bias=True)
      )
    )
    (PRM1): PRM1(
      (above_conv2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (above_bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (right_conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (right_bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fuse_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fuse_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (CAM): CAM(
        (softmax): Softmax(dim=-1)
        (conv1): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
      )
      (FAPA): FAPA(
        (softmax): Softmax(dim=-1)
        (FAPAEnc): FAPAEnc(
          (pool1): DAP(
            (conv1x1): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool2): DAP(
            (conv1x1): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool3): DAP(
            (conv1x1): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (pool4): DAP(
            (conv1x1): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (sigmoid): Sigmoid()
          )
          (conv1): Sequential(
            (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv2): Sequential(
            (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv3): Sequential(
            (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (conv4): Sequential(
            (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
        )
        (conv_query): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        (conv_key): Linear(in_features=128, out_features=32, bias=True)
        (conv_value): Linear(in_features=128, out_features=128, bias=True)
      )
    )
    (linearr1): Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (linearr2): Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (linearr3): Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (linearr4): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (cov1x1_1): Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1))
    (cov1x1_2): Conv2d(192, 512, kernel_size=(1, 1), stride=(1, 1))
    (cov1x1_3): Conv2d(384, 1024, kernel_size=(1, 1), stride=(1, 1))
    (cov1x1_4): Conv2d(768, 2048, kernel_size=(1, 1), stride=(1, 1))
    (linearrpred): Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (final_up): Final_PatchExpand2D(
      (expand): Linear(in_features=96, out_features=384, bias=False)
      (norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
    )
    (final_conv): Conv2d(24, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)