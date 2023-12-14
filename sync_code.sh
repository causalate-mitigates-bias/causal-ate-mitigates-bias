# run this file as "cmd < sync_code.sh" in cmd
# or "./sync_code.sh" on any linux based terminal.

# Switch to the right profile

export KEY="causalatemitigatesbias"


git config --local user.name "CAte Mitigates Bias"
git config --local user.email  "causalatemitigatesbias@gmail.com"

# Add the correct key to the github account
git config --add --local core.sshCommand 'ssh -i ~/.ssh/$KEY'


# Add all changes, commit with the provided message, and push
git add -A
# Use single quotes and not double quote. Why?
git commit -m "updates"
git commit -m 'updates'
git push --set-upstream origin main
