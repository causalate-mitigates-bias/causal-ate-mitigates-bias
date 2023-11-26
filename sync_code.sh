::# run this file as "cmd < sync_code.sh update"


# Extract the commit message from args[0]
commit_message="$0"
echo "commit_message is" commit_message

# Add all changes, commit with the provided message, and push
git add -A
git commit -m "$commit_message"
git push