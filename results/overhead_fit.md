Overhead per sublayer: 37.8 us
Effective weight bandwidth: 1021 GB/s

| Weights | Loops | Measured ms/token | Fitted ms/token | Overhead share | GB read/token | Bandwidth ceiling tok/s |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 1 | 9.98 | 9.94 | 27% | 7.37 | 137 |
| mxfp8 | 1 | 6.96 | 6.81 | 40% | 4.17 | 242 |
| nvfp4 | 1 | 5.29 | 5.29 | 51% | 2.62 | 384 |
| bf16 | 2 | 15.08 | 15.10 | 28% | 11.09 | 91 |
| mxfp8 | 2 | 10.39 | 10.22 | 41% | 6.12 | 165 |
| nvfp4 | 2 | 7.79 | 7.86 | 54% | 3.71 | 272 |
| bf16 | 4 | 25.29 | 25.41 | 29% | 18.54 | 54 |
| mxfp8 | 4 | 17.26 | 17.05 | 43% | 10.01 | 101 |
| nvfp4 | 4 | 12.81 | 13.01 | 56% | 5.88 | 172 |
