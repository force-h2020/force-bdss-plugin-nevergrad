@echo off

REM Windows native script to evaluate multiple points on a workflow.
REM %~f1      text file with parameters (tab or comma delimited), one point per line.
REM %~f2      json file with workflow
REM %~f3      text file with KPIs(tab delimited), one point per line.
REM example: evalf input.txt gaussian.json output.txt

REM delete the current KPIs (content of the output file)
break > %~f3

REM loop through the points to be evaluated (lines in the input file)
FOR /F "tokens=*" %%A in (%~f1) do  (
   echo %%A | force_bdss --evaluate %~f2 >> %~f3
)
