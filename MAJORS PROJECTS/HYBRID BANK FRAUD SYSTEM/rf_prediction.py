import joblib
import pandas as pd

# ---------------- Load trained ML model ----------------
model = joblib.load(r"rf_model.joblib")

# ---------------- User Input ----------------
print("\nEnter Transaction Details")

step = int(input("Step: "))
amount = float(input("Amount: "))
oldOrg = float(input("Sender Old Balance: "))
newOrg = float(input("Sender New Balance: "))
oldDest = float(input("Receiver Old Balance: "))
newDest = float(input("Receiver New Balance: "))
tx_type = input("Transaction Type (CASH_OUT / TRANSFER / PAYMENT / DEBIT): ").upper()

# ---------------- HARD RULE ENGINE (LOGIC) ----------------
def hard_rules(amount, oldOrg, newOrg, oldDest, newDest):
    if amount > 0 and (oldOrg - newOrg) < 0.95 * amount:
        return True, "Sender balance not deducted"

    if (newDest - oldDest) > 0 and oldOrg == newOrg:
        return True, "Receiver credited without sender debit"

    if newOrg < 0 or newDest < 0:
        return True, "Negative balance detected"

    return False, None

is_fraud_rule, reason = hard_rules(amount, oldOrg, newOrg, oldDest, newDest)

if is_fraud_rule:
    print("\n🚫 FRAUD — BLOCK PAYMENT")
    print("Reason:", reason)
    exit()

# ---------------- FEATURE ENGINEERING ----------------
balance_diff_orig = oldOrg - newOrg
balance_diff_dest = oldDest - newDest

data = {
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldOrg,
    "newbalanceOrig": newOrg,
    "oldbalanceDest": oldDest,
    "newbalanceDest": newDest,
    "balance_diff_orig": balance_diff_orig,
    "balance_diff_dest": balance_diff_dest,
    "type": tx_type
}

df = pd.DataFrame([data])

# ---------------- One-Hot Encode ----------------
df = pd.get_dummies(df, columns=["type"], drop_first=True)
df = df.reindex(columns=model.feature_names_in_, fill_value=0)

# ---------------- ML RISK SCORING ----------------
fraud_prob = model.predict_proba(df)[0][1]

LOW_RISK = 0.30
HIGH_RISK = 0.70

print("\n----- FINAL DECISION -----")
print(f"Fraud Probability: {fraud_prob:.4f}")

if fraud_prob < LOW_RISK:
    print("✅ NOT FRAUD — Allow Transaction")

elif fraud_prob < HIGH_RISK:
    print("⚠️ FLAGGED — Manual Review / OTP Required")

else:
    print("🚫 FRAUD — Block Payment (ML High Risk)")
