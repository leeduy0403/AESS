import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import {
  Accordion,
  AccordionSummary,
  Button,
  FormControl,
  MenuItem,
  Select,
  Typography,
} from "@mui/material";
import moment from "moment";
import { Avatar, Modal } from "flowbite-react";
import { HiOutlineExclamationCircle } from "react-icons/hi";
import { useSelector } from "react-redux";
import {
  Assignment as AssignmentIcon,
  ExpandMore as ExpandMoreIcon,
  Groups as GroupsIcon,
  Person as PersonIcon,
  RemoveCircle as RemoveCircleIcon,
} from "@mui/icons-material";

export default function ClassGroup() {
  const { classId } = useParams();
  const { currentUser } = useSelector((state) => state.user);
  const location = useLocation();
  const navigate = useNavigate();
  const [assignments, setAssignments] = useState([]);
  const [assignmentId, setAssignmentId] = useState("");
  const [assignment, setAssignment] = useState([]);
  const [ungroupedStudents, setUngroupedStudents] = useState([]);
  const [numberOfStudents, setNumberOfStudents] = useState(0);
  const [showModalJoinGroup, setShowModalJoinGroup] = useState(false);
  const [showModalLeaveGroup, setShowModalLeaveGroup] = useState(false);
  const [showModalRemoveMember, setShowModalRemoveMember] = useState(false);
  const [showModalDeleteGroup, setShowModalDeleteGroup] = useState(false);
  const [groupIdToJoin, setGroupIdToJoin] = useState("");
  const [groupIdToLeave, setGroupIdToLeave] = useState("");
  const [userIdToDelete, setUserIdToDelete] = useState("");
  const [groupIdToRemoveMember, setGroupIdToRemoveMember] = useState("");
  const [groupIdToDelete, setGroupIdToDelete] = useState("");

  const fetchAssignment = async () => {
    if (assignmentId.length === 0) {
      return;
    }
    try {
      const resAssignment = await fetch(
        `/api/assignment/get/${classId}?assignmentId=${assignmentId}`
      );
      const assignment = await resAssignment.json();
      if (resAssignment.ok) {
        setAssignment(assignment.assignments[0]);
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const fetchUngroupedMembers = async () => {
    if (assignmentId.length === 0) {
      return;
    }
    try {
      const res = await fetch(
        `/api/assignment/get-ungrouped-students/${assignmentId}`
      );
      const data = await res.json();
      if (res.ok) {
        setUngroupedStudents(data.unGroupedStudents);
        setNumberOfStudents(data.numberOfStudents);
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  useEffect(() => {
    fetchAssignment();
    fetchUngroupedMembers();
  }, [classId, assignmentId]);

  useEffect(() => {
    const fetchAssignments = async () => {
      try {
        const res = await fetch(`/api/assignment/get/${classId}`);
        const data = await res.json();
        if (res.ok) {
          setAssignments(data.assignments);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchAssignments();
  }, [classId]);

  const handleChange = (e) => {
    setAssignmentId(e.target.value);
    const urlParams = new URLSearchParams(location.search);
    urlParams.set("assignmentId", e.target.value);
    const searchQuery = urlParams.toString();
    navigate(`/class/${classId}?${searchQuery}`);
  };

  const handleJoinGroup = async () => {
    setShowModalJoinGroup(false);
    try {
      const res = await fetch(`/api/group/join/${assignmentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          groupId: groupIdToJoin,
          userId: currentUser?._id,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        fetchAssignment();
        fetchUngroupedMembers();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleLeaveGroup = async () => {
    setShowModalLeaveGroup(false);
    try {
      const res = await fetch(`/api/group/leave/${assignmentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          groupId: groupIdToLeave,
          userId: currentUser?._id,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        fetchAssignment();
        fetchUngroupedMembers();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleRemoveMember = async () => {
    setShowModalRemoveMember(false);
    try {
      const res = await fetch(
        `/api/group/remove-member/${classId}/${assignmentId}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            groupId: groupIdToRemoveMember,
            userId: userIdToDelete,
          }),
        }
      );
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        fetchAssignment();
        fetchUngroupedMembers();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleDeleteGroup = async () => {
    setShowModalDeleteGroup(false);
    try {
      const res = await fetch(
        `/api/group/delete/${assignmentId}/${groupIdToDelete}`,
        {
          method: "DELETE",
        }
      );
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        fetchAssignment();
        fetchUngroupedMembers();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <div className="font-bold text-lg">Choose your division:</div>
        <form>
          <FormControl fullWidth>
            <Select
              value={assignmentId}
              onChange={handleChange}
              displayEmpty
              inputProps={{ "aria-label": "Without label" }}
            >
              <MenuItem value="">Select an assignment</MenuItem>
              {assignments?.length > 0 &&
                assignments.map((assignment, index) => (
                  <MenuItem value={assignment?._id} key={index}>
                    {assignment?.title}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </form>
      </div>
      <Accordion
        defaultExpanded
        className="mb-4"
        sx={{
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: "white" }} />}
          aria-controls="panel1-content"
          id="panel1-header"
          sx={{
            backgroundColor: "#26597C",
            color: "#ffffff",
            borderTopLeftRadius: 12,
            borderTopRightRadius: 12,
          }}
        >
          <div className="flex gap-2 items-center">
            <AssignmentIcon />
            <Typography
              component="span"
              style={{ fontSize: "18px", fontWeight: "bold" }}
            >
              {assignment?.title}
            </Typography>
            {assignment?.type === "Individual" ? (
              <PersonIcon />
            ) : (
              <GroupsIcon />
            )}
          </div>
        </AccordionSummary>
        <div style={{ backgroundColor: "#F8F8D5", padding: "4px" }}>
          <div className="flex gap-2 my-5 lg:w-11/12 mx-auto">
            <div className="flex flex-col gap-2">
              <div className="font-bold">Open Date: </div>
              <div className="font-bold">Due Date: </div>
              <div className="font-bold">Description: </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="font-bold text-red-600">
                {assignment?.startDate
                  ? moment(assignment?.startDate).format("HH:mm:ss DD/MM/YYYY")
                  : "---"}
              </div>
              <div className="font-bold text-red-600">
                {assignment?.endDate
                  ? moment(assignment?.endDate).format("HH:mm:ss DD/MM/YYYY")
                  : "---"}
              </div>
              <div>{assignment?.description || "---"}</div>
            </div>
          </div>
        </div>
      </Accordion>
      {assignmentId && assignment?.type === "Group" && (
        <div className="mb-4">
          <Link
            to={`/class/${classId}/add-group?assignmentId=${assignment?._id}`}
          >
            <Button
              variant="contained"
              component="label"
              style={{
                backgroundColor: "#26597C",
                color: "#ffffff",
                textTransform: "none",
              }}
              size="large"
            >
              Create New Group
            </Button>
          </Link>
        </div>
      )}
      {currentUser?.isStudent
        ? assignmentId && (
            <div className="overflow-x-auto shadow-md rounded-lg">
              <table className="w-full text-sm bg-white border border-gray-300">
                <thead className="text-white bg-[#26597C]">
                  <tr className="text-left">
                    <th className="p-4 border">Group</th>
                    <th className="p-4 border">Members</th>
                    <th className="p-4 border">Member count</th>
                    <th className="p-4 border">Join / Leave</th>
                  </tr>
                </thead>
                <tbody>
                  {assignment?.groups?.length > 0 &&
                    assignment?.groups.map((group, index) => (
                      <tr
                        key={index}
                        className={`transition-all ${
                          index % 2 === 0 ? "bg-white" : "bg-[#F8F8D5]"
                        }`}
                      >
                        <td className="p-4 border border-gray-300 font-medium text-sky-700">
                          {group?.name}
                        </td>
                        <td className="p-4 border border-gray-300">
                          <div className="flex flex-col gap-4">
                            {group?.members?.length > 0 &&
                              group?.members.map((member, index) => (
                                <div
                                  key={index}
                                  className="flex gap-2 items-center justify-start"
                                >
                                  <Avatar
                                    alt="User avatar"
                                    img={member?.profilePicture}
                                    size="sm"
                                    rounded
                                  />
                                  <div>{member?.studentId}</div>
                                  <div className="text-cyan-600">
                                    {member?.name}
                                  </div>
                                </div>
                              ))}
                          </div>
                        </td>
                        <td className="p-4 border border-gray-300">
                          {group?.members?.length}/{assignment?.maxMemberGroup}
                        </td>
                        <td className="p-4 border border-gray-300">
                          <div>
                            {group?.members?.some(
                              (member) => member?._id === currentUser?._id
                            ) ? (
                              <Button
                                variant="contained"
                                component="label"
                                style={{
                                  backgroundColor: "oklch(0.577 0.245 27.325)",
                                  color: "#ffffff",
                                  textTransform: "none",
                                }}
                                fullWidth
                                onClick={(e) => {
                                  setShowModalLeaveGroup(true);
                                  setGroupIdToLeave(group?._id);
                                }}
                              >
                                Leave
                              </Button>
                            ) : group?.members?.length <
                              assignment?.maxMemberGroup ? (
                              <Button
                                variant="contained"
                                component="label"
                                style={{
                                  backgroundColor: "#26597C",
                                  color: "#ffffff",
                                  textTransform: "none",
                                }}
                                fullWidth
                                onClick={(e) => {
                                  setShowModalJoinGroup(true);
                                  setGroupIdToJoin(group?._id);
                                }}
                              >
                                Join
                              </Button>
                            ) : (
                              "Full"
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  {ungroupedStudents?.length > 0 && (
                    <tr className="transition-all bg-gray-100">
                      <td className="p-4 border border-gray-300 font-medium text-sky-700">
                        {assignment?.type === "Individual" && "All Students"}
                        {assignment?.type === "Group" && "Ungrouped Students"}
                      </td>
                      <td className="p-4 border border-gray-300">
                        <div className="flex flex-col gap-4">
                          {ungroupedStudents?.length > 0 &&
                            ungroupedStudents.map((member, index) => (
                              <div
                                key={index}
                                className="flex gap-2 items-center justify-start"
                              >
                                <Avatar
                                  alt="User avatar"
                                  img={member?.profilePicture}
                                  size="sm"
                                  rounded
                                />
                                <div>{member?.studentId}</div>
                                <div className="text-cyan-600">
                                  {member?.name}
                                </div>
                              </div>
                            ))}
                        </div>
                      </td>
                      <td className="p-4 border border-gray-300">
                        {ungroupedStudents?.length}/{numberOfStudents}
                      </td>
                      <td className="p-4 border border-gray-300"></td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )
        : assignmentId && (
            <div className="overflow-x-auto shadow-md rounded-lg">
              <table className="w-full text-sm bg-white border border-gray-300">
                <thead className="text-white bg-[#26597C]">
                  <tr className="text-left">
                    <th className="p-4 border">Group</th>
                    <th className="p-4 border">Members</th>
                    <th className="p-4 border">Member count</th>
                    <th className="p-4 border">Delete</th>
                  </tr>
                </thead>
                <tbody>
                  {assignment?.groups?.length > 0 &&
                    assignment?.groups.map((group, index) => (
                      <tr
                        key={index}
                        className={`transition-all ${
                          index % 2 === 0 ? "bg-white" : "bg-[#F8F8D5]"
                        }`}
                      >
                        <td className="p-4 border border-gray-300 font-medium text-sky-700">
                          {group?.name}
                        </td>
                        <td className="p-4 border border-gray-300">
                          <div className="flex flex-col gap-4">
                            {group?.members?.length > 0 &&
                              group?.members.map((member, index) => (
                                <div
                                  key={index}
                                  className="flex items-center justify-between"
                                >
                                  <div className="flex gap-2 items-center justify-start">
                                    <Avatar
                                      alt="User avatar"
                                      img={member?.profilePicture}
                                      size="sm"
                                      rounded
                                    />
                                    <div>{member?.studentId}</div>
                                    <Link
                                      to={`/class/${classId}/view-attempts?assignmentId=${assignmentId}&studentId=${member?._id}`}
                                      className="hover:underline text-cyan-600"
                                    >
                                      {member?.name}
                                    </Link>
                                  </div>
                                  <RemoveCircleIcon
                                    key={index}
                                    color="error"
                                    sx={{ "&:hover": { cursor: "pointer" } }}
                                    fontSize="small"
                                    onClick={(e) => {
                                      setShowModalRemoveMember(true);
                                      setGroupIdToRemoveMember(group);
                                      setUserIdToDelete(member?._id);
                                    }}
                                  />
                                </div>
                              ))}
                          </div>
                        </td>
                        <td className="p-4 border border-gray-300">
                          {group?.members?.length}/{assignment?.maxMemberGroup}
                        </td>
                        <td className="px-4 py-2 border border-gray-300">
                          <div className="flex flex-col my-2 gap-2 justify-center">
                            {/* <Button
                              variant="contained"
                              component="label"
                              style={{
                                backgroundColor: "#26597C",
                                color: "#ffffff",
                                textTransform: "none",
                              }}
                              fullWidth
                              onClick={(e) => {
                                setShowModalJoinGroup(true);
                                setGroupIdToJoin(group?._id);
                              }}
                            >
                              Edit group
                            </Button> */}
                            <Button
                              variant="contained"
                              component="label"
                              style={{
                                backgroundColor: "oklch(0.577 0.245 27.325)",
                                color: "#ffffff",
                                textTransform: "none",
                              }}
                              fullWidth
                              onClick={(e) => {
                                setShowModalDeleteGroup(true);
                                setGroupIdToDelete(group?._id);
                              }}
                            >
                              Delete group
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  {ungroupedStudents?.length > 0 && (
                    <tr className="transition-all bg-gray-100">
                      <td className="p-4 border border-gray-300 font-medium text-sky-700">
                        {assignment?.type === "Individual" && "All Students"}
                        {assignment?.type === "Group" && "Ungrouped Students"}
                      </td>
                      <td className="p-4 border border-gray-300">
                        <div className="flex flex-col gap-4">
                          {ungroupedStudents?.length > 0 &&
                            ungroupedStudents.map((member, index) => (
                              <div
                                key={index}
                                className="flex gap-2 items-center justify-start"
                              >
                                <Avatar
                                  alt="User avatar"
                                  img={member?.profilePicture}
                                  size="sm"
                                  rounded
                                />
                                <div>{member?.studentId}</div>
                                <div className="text-cyan-600">
                                  {member?.name}
                                </div>
                              </div>
                            ))}
                        </div>
                      </td>
                      <td className="p-4 border border-gray-300">
                        {ungroupedStudents?.length}/{numberOfStudents}
                      </td>
                      <td className="p-4 border border-gray-300"></td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
      <Modal
        show={showModalJoinGroup}
        onClose={() => setShowModalJoinGroup(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to join this group?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleJoinGroup}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => setShowModalJoinGroup(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalLeaveGroup}
        onClose={() => setShowModalLeaveGroup(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to leave this group?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleLeaveGroup}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => setShowModalLeaveGroup(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalRemoveMember}
        onClose={() => setShowModalRemoveMember(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to remove this member?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleRemoveMember}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => setShowModalRemoveMember(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalDeleteGroup}
        onClose={() => setShowModalDeleteGroup(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to delete this group?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleDeleteGroup}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => setShowModalDeleteGroup(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
}
