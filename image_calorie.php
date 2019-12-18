<?php
include_once 'project_connect.php';
$servername = "localhost";
$image_id ="image_id";
$fat_calorie = "fat_calorie";
$carb_calorie = "carb_calorie";
$protein_calorie = "protein_calorie";
$total_calories = "total_calories";
$user_id = "user_id";

$sql="select*from image_calorie ";
$r=mysqli_query($conn,$sql);
$arr=array();
while($row=mysqli_fetch_assoc($r)){
    $arr[]=$row;
} 
header('Connect-Type:application/json');
echo json_encode($arr);

