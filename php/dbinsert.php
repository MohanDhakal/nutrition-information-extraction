<?php
include 'project_connect.php';

if (isset($_POST['UserName'])) {
    $UserName =$_POST['UserName'];
}

if (isset($_POST['fId'])) {
    $fId = $_POST['fId'];
}
if (isset($_POST['report'])) {
    $report = $_POST['report'];
}


    $sql="Insert into user(UserName,UId,fId,report)values('{$UserName}','{$UId}','{$fId}','{$report}');";


$r=mysqli_query($conn,$sql);
$array=array();
$result="user inserted";
$array[]=$result;
header('Content-Type:application/json');
echo json_encode($array);
?>