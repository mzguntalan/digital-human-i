# Digital Human I

Lecture 2025年 04月 21日

## Dataset

The FDDB Dataset can be obtained on Kaggle via this [link](https://paperswithcode.com/dataset/fddb). Save the files in the `./fddb/` directory such that the annotations are in `./fddb/FDDB-folds/` and the images are in `./fddb/images/`.

## Code

The code used to generate the report are in report.

### Running the Code

To run the evaluation, run either of the following.

```bash
python report/run_eval_cnn.py
python report/run_eval_hog.py
```

To run the visualization, run either of the following.

```bash
python report/run_viz_cnn.py
python report/run_viz_hog.py
```

To run on the hand image, run either of the following.

```python
python cnn_face_detection.py -i images/hand_drawn.jpg
python hog_face_detection.py -i images/hand_drawn.jpg
```

## Environment

Use the provided `requirements.txt` to setup the environment.
