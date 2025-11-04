import { Avatar, Dropdown, Navbar } from "flowbite-react";
import { Link, useLocation } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { signoutSuccess } from "../redux/user/userSlice";
import LMS from "../assets/LMS.png";
import { Button } from "@mui/material";
import { closeChat } from "@/redux/chat/chatSlice";

export default function Header() {
  const path = useLocation().pathname;
  const dispatch = useDispatch();
  const { currentUser } = useSelector((state) => state.user);

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
        dispatch(closeChat());
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  return (
    <Navbar className="border-b-2">
      <Link
        to={"/"}
        className="self-center whitespace-nowrap text-sm sm:text-xl font-semibold dark:text-white"
      >
        <img src={LMS} alt="logo" className="w-[85px]" />
      </Link>
      <div className="flex gap-2 md:order-2 items-center z-20">
        {currentUser ? (
          <Dropdown
            arrowIcon={false}
            inline
            label={
              <Avatar
                alt="User avatar"
                img={currentUser?.profilePicture}
                rounded
              />
            }
          >
            <Dropdown.Header>
              <span className="block text-sm font-medium truncate">
                {currentUser?.email}
              </span>
            </Dropdown.Header>
            <Link to={"/dashboard?tab=profile"}>
              <Dropdown.Item>Profile</Dropdown.Item>
            </Link>
            <Dropdown.Divider />
            <Dropdown.Item onClick={handleSignout}>Sign Out</Dropdown.Item>
          </Dropdown>
        ) : (
          <Link to={"/sign-in"}>
            <Button
              variant="contained"
              style={{
                backgroundColor: "#26597C",
              }}
            >
              Sign In
            </Button>
          </Link>
        )}
        <Navbar.Toggle />
      </div>
      <Navbar.Collapse>
        <Navbar.Link active={path === "/"} as={"div"}>
          <Link to={"/"} className="text-xl">
            Home
          </Link>
        </Navbar.Link>
        <Navbar.Link active={path === "/classes"} as={"div"}>
          <Link to={"/classes"} className="text-xl">
            Course
          </Link>
        </Navbar.Link>
        <Navbar.Link active={path === "/chat"} as={"div"}>
          <Link to={"/chat"} className="text-xl">
            Chat
          </Link>
        </Navbar.Link>
        <Navbar.Link active={path === "/calendar"} as={"div"}>
          <Link to={"/calendar"} className="text-xl">
            Calendar
          </Link>
        </Navbar.Link>
        <Navbar.Link active={path === "/dashboard"} as={"div"}>
          <Link to={"/dashboard"} className="text-xl">
            Settings
          </Link>
        </Navbar.Link>
      </Navbar.Collapse>
    </Navbar>
  );
}
