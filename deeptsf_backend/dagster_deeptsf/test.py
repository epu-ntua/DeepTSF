import requests

DAGSTER_URL = "https://deeptsf-dagster.energy-guard.eu/graphql"
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI0aV9ZY3FpMXhrQml4T21mZGYtVTA1Zl9ET0E3MXpiNmk2VVB3NWp4amhzIn0.eyJleHAiOjE3NzEyNTUxNzEsImlhdCI6MTc3MTI1NDg3MSwianRpIjoiZTFjYTIzOTYtNjYxNC00NmJmLWEyMzEtOWFmZjAwMDQyMDNiIiwiaXNzIjoiaHR0cHM6Ly9rZXljbG9hay50b29sYm94LmVwdS5udHVhLmdyL3JlYWxtcy9FbmVyZ3lHdWFyZCIsImF1ZCI6WyJtbGZsb3ctZW5lcmd5Z3VhcmQiLCJhY2NvdW50Il0sInN1YiI6IjllNTRmZTU5LTFkZWUtNGRiNC05MWY1LTIyOTIwZjU2ZmUzZiIsInR5cCI6IkJlYXJlciIsImF6cCI6Im1sZmxvdy1lbmVyZ3lndWFyZCIsInNlc3Npb25fc3RhdGUiOiJmNWVkZGQxNy1hM2ZjLTQ2NzUtYmNiYS0wMGFhM2MzMjhkMjQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbImh0dHBzOi8vbWxmbG93LmVuZXJneS1ndWFyZC5ldS8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtZW5lcmd5Z3VhcmQiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoiZ3JvdXBzIGVtYWlsIHByb2ZpbGUgb3BlbmlkIiwic2lkIjoiZjVlZGRkMTctYTNmYy00Njc1LWJjYmEtMDBhYTNjMzI4ZDI0IiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJUaGVvZG9zaW9zIFBvdW50cmlkaXMiLCJncm91cHMiOlsiaWNjcyIsIm1sZmxvdy11c2VycyJdLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJkZW1vX2RlZXB0c2ZfdXNlckB5YWhvby5jb20iLCJnaXZlbl9uYW1lIjoiVGhlb2Rvc2lvcyIsImZhbWlseV9uYW1lIjoiUG91bnRyaWRpcyIsImVtYWlsIjoiZGVtb19kZWVwdHNmX3VzZXJAeWFob28uY29tIn0.fpAAEMg6Xuv65HgTq0tbv6CbtOyfCb6hexbXH-po4hdUa-XRjfPCF0Jdcp-xZ4n4mbn5weCUZh2Asz1Y_L3Tnkh1x5CkSZdDgPDQ_UAacgBOtnb_N3XmPVX0IOL9-DQ40lCysqnOsYGwUXCJCmLKj3Wf2xg_ZIuE5eszo9-6NhNKX8--U-E5p8LvDVULVU3_Rfn6T_jtYGFAjcpMiDKcSE2krUzrIAa5virj5tIxqWEaoGrv3V7YVce3sn1xcknzcBJUd1uGYX8HGZ0zzedOUyONETU9LMFhnwI2YqqiE8HlG7TdBOg7l4QHm3_ayMAok8Bzipg18ZFR-EJvX0gKgg"  # set to your Bearer token string if needed

query = """
query {
  repositoriesOrError {
    __typename
    ... on RepositoryConnection {
      nodes {
        name
        location { name }
        pipelines { name isJob }
      }
    }
    ... on PythonError { message }
  }
}
"""

headers = {"Content-Type": "application/json"}
if TOKEN:
    headers["Authorization"] = f"Bearer {TOKEN}"

r = requests.post(DAGSTER_URL, headers=headers, json={"query": query}, timeout=20)
print("HTTP", r.status_code)
print(r.text[:2000])  # <-- keep this if it errors

data = r.json()
nodes = data["data"]["repositoriesOrError"]["nodes"]

for n in nodes:
    loc = n["location"]["name"]
    repo = n["name"]
    for p in n["pipelines"]:
        if p["isJob"]:
            print(f"{loc} / {repo} / {p['name']}")
