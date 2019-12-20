<?php
$a=mysql_connect('localhost','root','','jd');
if(!$a)
	echo "not connected";
else
	echo "connected";
?>