::# run this file as "cmd < sync_code.sh update"

# Extract the commit message from args[1]
commit_message="$1"

echo $1

# Add all changes, commit with the provided message, and push
git add -A
git commit -m "$commit_message"
git push