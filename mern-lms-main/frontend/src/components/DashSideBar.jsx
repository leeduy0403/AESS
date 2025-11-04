import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Sidebar } from "flowbite-react";
import {
  HiUser,
  HiArrowSmRight,
  HiDocumentText,
  HiClipboardCopy,
} from "react-icons/hi";
import { PiStudentBold } from "react-icons/pi";
import { GiTeacher } from "react-icons/gi";
import { FaBuilding } from "react-icons/fa";
import { SiCoursera } from "react-icons/si";
import { FaUserTie } from "react-icons/fa6";
import { signoutSuccess } from "../redux/user/userSlice";
import { useDispatch, useSelector } from "react-redux";

export default function DashSideBar() {
  const location = useLocation();
  const dispatch = useDispatch();
  const { currentUser } = useSelector((state) => state.user);
  const [tab, setTab] = useState("");

  useEffect(() => {
    const urlParams = new URLSearchParams(location.search);
    const tabFromUrl = urlParams.get("tab");
    if (tabFromUrl) {
      setTab(tabFromUrl);
    }
  }, [location.search]);

  const handleSignout = async () => {
    try {
      const res = await fetch("/api/user/signout", {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        dispatch(signoutSuccess());
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  return (
    <Sidebar className="w-full md:w-60" color="blue">
      <Sidebar.Items>
        <Sidebar.ItemGroup className="flex flex-col gap-2">
          <Link to={"/dashboard?tab=profile"}>
            <Sidebar.Item
              active={tab === "profile"}
              icon={HiUser}
              label={currentUser.isAdmin ? "Admin" : "User"}
              labelColor="dark"
              as="div"
            >
              Profile
            </Sidebar.Item>
          </Link>
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=faculties"}>
              <Sidebar.Item
                active={tab === "faculties"}
                icon={FaBuilding}
                as="div"
              >
                Faculties
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=subjects"}>
              <Sidebar.Item
                active={tab === "subjects"}
                icon={HiDocumentText}
                as="div"
              >
                Subjects
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=courses"}>
              <Sidebar.Item
                active={tab === "courses"}
                icon={SiCoursera}
                as="div"
              >
                Courses
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=classes"}>
              <Sidebar.Item
                active={tab === "classes"}
                icon={GiTeacher}
                as="div"
              >
                Classes
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=educators"}>
              <Sidebar.Item
                active={tab === "educators"}
                icon={FaUserTie}
                as="div"
              >
                Educators
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=students"}>
              <Sidebar.Item
                active={tab === "students"}
                icon={PiStudentBold}
                as="div"
              >
                Students
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <div className="border-t-2 border-gray-300 pt-4">
              <Link to={"/dashboard?tab=create-user"}>
                <Sidebar.Item
                  active={tab === "create-user"}
                  icon={HiUser}
                  as="div"
                >
                  Create User
                </Sidebar.Item>
              </Link>
            </div>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=create-faculty"}>
              <Sidebar.Item
                active={tab === "create-faculty"}
                icon={FaBuilding}
                as="div"
              >
                Create Faculty
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=create-subject"}>
              <Sidebar.Item
                active={tab === "create-subject"}
                icon={HiDocumentText}
                as="div"
              >
                Create Subject
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=create-course"}>
              <Sidebar.Item
                active={tab === "create-course"}
                icon={SiCoursera}
                as="div"
              >
                Create Course
              </Sidebar.Item>
            </Link>
          )}
          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=create-class"}>
              <Sidebar.Item
                active={tab === "create-class"}
                icon={GiTeacher}
                as="div"
              >
                Create Class
              </Sidebar.Item>
            </Link>
          )}

          {currentUser?.isAdmin && (
            <Link to={"/dashboard?tab=assign-course"}>
              <Sidebar.Item
                active={tab === "assign-course"}
                icon={HiClipboardCopy}
                as="div"
              >
                Assign Course
              </Sidebar.Item>
            </Link>
          )}
          <div className="border-t-2 border-gray-300 pt-4">
            <Sidebar.Item
              icon={HiArrowSmRight}
              className="text-red-500 hover:bg-red-100 dark:hover:bg-red-900 cursor-pointer"
              onClick={handleSignout}
            >
              Sign Out
            </Sidebar.Item>
          </div>
        </Sidebar.ItemGroup>
      </Sidebar.Items>
    </Sidebar>
  );
}
