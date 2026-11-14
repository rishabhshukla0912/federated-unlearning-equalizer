# Client datasets

One folder per client, each holding `<client>_dataset.mat` (MATLAB v7.3, readable with `h5py`).

| Field | Meaning |
|---|---|
| `rxSymbols_all` | received frequency-domain symbols (complex) |
| `channelH_all` | channel frequency response (complex) |
| `txSymbols_all` | transmitted 256-QAM symbols (complex) |
| `allLabels` | symbol class labels, 0 to 255 |

Samples are stored frame by frame, 1024 subcarriers per frame. `fu_equalizer.data.read_frames` slices a frame range and returns arrays of shape (frames, 1024).
