@echo off

REM Windows native script to optimize a workflow.
REM %~f1      json file with workflow
REM %~f2      text file with KPIs(tab delimited), one point per line.
REM example: evalf input.txt gaussian.json output.txt

echo %%A | edm run -e force-py36 -- force_bdss %~f1 >> %~f2
