import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import DashSideBar from "../components/DashSideBar";
import DashProfile from "../components/DashProfile";
import DashCourses from "../components/DashCourses";
import DashEducators from "../components/DashEducators";
import DashStudents from "../components/DashStudents";
import DashCreateCourse from "../components/DashCreateCourse";
import SignUp from "../pages/SignUp";
import DashAssign from "../components/DashAssign";
import DashClasses from "@/components/DashClasses";
import DashSubjects from "@/components/DashSubjects";
import DashFaculties from "@/components/DashFaculties";
import DashCreateFaculty from "@/components/DashCreateFaculty";
import DashCreateSubject from "@/components/DashCreateSubject";
import DashCreateClass from "@/components/DashCreateClass";
import DashCreateUser from "@/components/DashCreateUser";

export default function Dashboard() {
  const location = useLocation();
  const [tab, setTab] = useState("");
  useEffect(() => {
    const urlParams = new URLSearchParams(location.search);
    const tabFromUrl = urlParams.get("tab");
    if (tabFromUrl) {
      setTab(tabFromUrl);
    }
  }, [location.search]);
  return (
    <div className="min-h-screen flex flex-col md:flex-row">
      <div className="md:w-56">
        <DashSideBar />
      </div>
      {tab === "profile" && <DashProfile />}
      {tab === "faculties" && <DashFaculties />}
      {tab === "subjects" && <DashSubjects />}
      {tab === "courses" && <DashCourses />}
      {tab === "classes" && <DashClasses />}
      {tab === "educators" && <DashEducators />}
      {tab === "students" && <DashStudents />}
      {tab === "create-faculty" && <DashCreateFaculty />}
      {tab === "create-subject" && <DashCreateSubject />}
      {tab === "create-course" && <DashCreateCourse />}
      {tab === "create-class" && <DashCreateClass />}
      {tab === "create-user" && <DashCreateUser />}
      {tab === "assign-course" && <DashAssign />}
    </div>
  );
}
