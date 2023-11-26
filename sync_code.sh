# run this file as "cmd < sync_code.sh"

# Add all changes, commit with the provided message, and push
git add -A
# Use single quotes and not double quote. Why?
git commit -m "updates"
git push --set-upstream origin main
