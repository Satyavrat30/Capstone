# House Predictor (FastAPI Pattern)

This follows the exact structure pattern from your screenshot:

```text
/my-capstone
  model.pkl
  main.py
  requirements.txt
  /static
    index.html
    script.js
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

Open: `http://127.0.0.1:8000`

## Notes

- `model.pkl` is auto-created on first run if not present.
- Frontend uses `fetch()` in `static/script.js` to call `/predict`.
