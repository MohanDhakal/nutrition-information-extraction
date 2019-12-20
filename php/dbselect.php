<?php
include_once 'project_connect.php';
$sql="select*from foods";
$r=mysqli_query($conn,$sql);
$arr=array();
while($row=mysqli_fetch_assoc($r)){
    $arr[]=$row;
} 
header('Connect-Type:application/json');
echo json_encode($arr);