# Project Status

## What problem does this project solve?

It demonstrates batch fraud screening for credit-card transactions using a saved Random Forest classifier.

## What is working?

- trained Random Forest artifact
- saved scaler
- Streamlit CSV upload
- schema validation
- batch prediction
- fraud-flag summary
- downloadable scored output
- deployment configuration

## How do I run it?

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Is it deployed?

Deployment is prepared but the live URL should only be recorded after a verified Render deployment.

## Does it use an API?

No. The app loads local serialized model artifacts and scores uploaded CSV rows. No external API keys or services are required.

## What changed in the September 2026 portfolio cleanup?

- replaced the unusable full-Anaconda `requirements.txt` export with minimal runtime dependencies
- added strict model input schema checks
- added missing-value and numeric validation
- added a downloadable input template
- added summary metrics and scored-output download
- added an explicit educational-use limitation
- documented the absence of an external API
- added Render configuration
- rewrote the README to reflect the app that actually exists

## Important limitation

The existing model/scaler are serialized artifacts from the original environment. Hosted deployment must be tested after dependency installation because scikit-learn pickle compatibility can vary across versions.

## Next step

Verify a clean hosted launch. If artifact compatibility fails, reproduce training in a pinned environment and export a fresh pipeline artifact rather than hiding the failure.
