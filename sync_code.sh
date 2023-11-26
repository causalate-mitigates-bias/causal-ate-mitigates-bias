::# run this file as "cmd < sync_code.sh"

# Add all changes, commit with the provided message, and push
git add -A
sleep 1
git commit -m "updates"
sleep 1

git push --set-upstream origin main