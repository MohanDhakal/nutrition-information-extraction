<?php
include_once 'project_connect.php';

$fat_calorie = 1.0;
$carb_calorie =2.0 ;
$protein_calorie = 3.0;
$total_calories =6.0;
$user_id = 1;

if(isset($_POST["fat_calorie"])){
    $fat_calorie = $_POST["fat_calorie"];

}
if(isset($_POST['protein_calorie'])){
    $protein_calorie = $_POST['protein_calorie'];   
}
if(isset($_POST['carb_calorie'])){
    $carb_calorie = $_POST['carb_calorie'];  
}

if(isset( $_POST['user_id'])){
    $user_id = $_POST['user_id'];  
}

$total_calories =$fat_calorie + $carb_calorie + $protein_calorie;
$mero_sql ="Insert Into image_calories(fat_calorie,carb_calorie,protein_calorie,calorie_t,user_id)VALUES('$fat_calorie','$carb_calorie','$protein_calorie','$total_calories','$user_id');";

mysqli_query($conn,$mero_sql);

if ($conn->query($mero_sql) === TRUE) {
    echo "New record created successfully";
} else {
    echo "Error: " . $mero_sql . "<br>" . $conn->error;
}

mysqli_close($conn);
?>
