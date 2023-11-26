::# run this file as "cmd < sync_code.sh update"

# Add all changes, commit with the provided message, and push
git add -A
git commit -m "updates"
sleep 2

git push --set-upstream origin main