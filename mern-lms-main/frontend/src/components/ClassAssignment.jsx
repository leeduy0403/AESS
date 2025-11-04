import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import Typography from "@mui/material/Typography";
import { useSelector } from "react-redux";
import { useEffect, useState } from "react";
import moment from "moment";
import { Link, useParams } from "react-router-dom";
import { Button } from "@mui/material";
import { Modal } from "flowbite-react";
import { HiOutlineExclamationCircle } from "react-icons/hi";
import pdf from "../assets/pdf.png";
import {
  Assignment as AssignmentIcon,
  ExpandMore as ExpandMoreIcon,
  Groups as GroupsIcon,
  Person as PersonIcon,
} from "@mui/icons-material";

export default function ClassAssignment() {
  const { classId } = useParams();
  const { currentUser } = useSelector((state) => state.user);
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const [assignments, setAssignments] = useState([]);
  const [showModalDeleteAssignment, setShowModalDeleteAssignment] =
    useState(false);
  const [showModalDeleteItem, setShowModalDeleteItem] = useState(false);
  const [
    showModalUpdateAssignmentIsHiddenTrue,
    setShowModalUpdateAssignmentIsHiddenTrue,
  ] = useState(false);
  const [
    showModalUpdateAssignmentIsHiddenFalse,
    setShowModalUpdateAssignmentIsHiddenFalse,
  ] = useState(false);
  const [showModalUpdateItemIsHiddenTrue, setShowModalUpdateItemIsHiddenTrue] =
    useState(false);
  const [
    showModalUpdateItemIsHiddenFalse,
    setShowModalUpdateItemIsHiddenFalse,
  ] = useState(false);
  const [assignmentToUpdate, setAssignmentToUpdate] = useState("");
  const [assignmentToDelete, setAssignmentToDelete] = useState("");
  const [itemToUpdate, setItemToUpdate] = useState("");
  const [itemToDelete, setItemToDelete] = useState("");
  const [assignmentIdOfItemToDelete, setAssignmentIdOfItemToDelete] =
    useState("");

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

  const handleUpdateAssignmentIsHiddenTrue = async () => {
    setShowModalUpdateAssignmentIsHiddenTrue(false);
    try {
      const res = await fetch(
        `/api/assignment/update/${classId}/${assignmentToUpdate}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            isHidden: true,
          }),
        }
      );
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        window.location.reload();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleUpdateAssignmentIsHiddenFalse = async () => {
    setShowModalUpdateAssignmentIsHiddenFalse(false);
    try {
      const res = await fetch(
        `/api/assignment/update/${classId}/${assignmentToUpdate}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            isHidden: false,
          }),
        }
      );
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        window.location.reload();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleUpdateItemIsHiddenTrue = async () => {
    setShowModalUpdateItemIsHiddenTrue(false);
    try {
      const res = await fetch(`/api/material/update/${itemToUpdate}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          isHidden: true,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        window.location.reload();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleUpdateItemIsHiddenFalse = async () => {
    setShowModalUpdateItemIsHiddenFalse(false);
    try {
      const res = await fetch(`/api/material/update/${itemToUpdate}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          isHidden: false,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        window.location.reload();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleDeleteAssignment = async () => {
    setShowModalDeleteAssignment(false);
    try {
      const res = await fetch(
        `/api/assignment/delete/${classId}/${assignmentToDelete}`,
        {
          method: "DELETE",
        }
      );
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        window.location.reload();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleDeleteItem = async () => {
    setShowModalDeleteItem(false);
    try {
      const res = await fetch(
        `/api/material/delete-material-assignment/${assignmentIdOfItemToDelete}/${itemToDelete}`,
        {
          method: "DELETE",
        }
      );
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        window.location.reload();
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  return (
    <div>
      {assignments?.length > 0 &&
        assignments.map((assignment, index) =>
          currentUser?.isStudent ? (
            !assignment?.isHidden && (
              <Accordion
                defaultExpanded
                className="mb-4"
                key={index}
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
                <div className="flex justify-between my-5 lg:w-11/12 mx-auto">
                  <div className="flex flex-col gap-2">
                    <div className="flex gap-2">
                      <div className="flex flex-col gap-2">
                        <div className="font-bold">Open date:</div>
                        <div className="font-bold">Due date:</div>
                        <div className="font-bold">Description:</div>
                      </div>
                      <div className="flex flex-col gap-2">
                        <div className="font-bold text-red-600">
                          {assignment?.startDate
                            ? moment(assignment?.startDate).format(
                                "HH:mm:ss DD/MM/YYYY"
                              )
                            : "---"}
                        </div>
                        <div className="font-bold text-red-600">
                          {assignment?.endDate
                            ? moment(assignment?.endDate).format(
                                "HH:mm:ss DD/MM/YYYY"
                              )
                            : "---"}
                        </div>
                        <div className="">
                          {assignment?.description || "---"}
                        </div>
                      </div>
                    </div>
                  </div>
                  <Link
                    to={`/class/${classId}/view-attempts?assignmentId=${assignment?._id}`}
                  >
                    <Button
                      variant="contained"
                      component="label"
                      style={{
                        backgroundColor: "#F8F8D5",
                        textTransform: "none",
                        color: "#000000",
                        border: "1px solid",
                      }}
                      size="large"
                    >
                      Submit
                    </Button>
                  </Link>
                </div>
                {assignment?.materials?.length > 0 &&
                  assignment?.materials.map((material, i) => (
                    <div
                      className="w-11/12 mx-auto border-2 border-gray-300 shadow-md p-4 mb-5 rounded-xl bg-gray-100 hover:bg-sky-100 transition"
                      key={i}
                    >
                      {!material?.isHidden &&
                        material?.materialUrls?.length > 0 &&
                        material?.materialUrls.map((url, j, arr) => (
                          <div key={j} className="flex flex-col gap-4 pt-4">
                            <div className="flex items-center justify-between lg:w-11/12 mx-auto">
                              <div className="flex items-center gap-4">
                                <img
                                  src={pdf}
                                  alt="pdf icon"
                                  className="w-6 h-6"
                                />
                                <Link
                                  to={url}
                                  target="_blank"
                                  className="hover:underline text-cyan-600 ml-1"
                                >
                                  {material?.nameFiles[j]?.substring(
                                    0,
                                    material?.nameFiles[j]?.lastIndexOf(".")
                                  )}
                                </Link>
                              </div>
                              {j === 0 && (
                                <div className="text-xs">
                                  Upload:{" "}
                                  {material?.createdAt
                                    ? moment(material?.createdAt).format(
                                        "DD/MM/YYYY"
                                      )
                                    : "---"}
                                </div>
                              )}
                            </div>
                            {j === arr.length - 1 && (
                              <div className="border-t border-black lg:w-11/12 mx-auto">
                                <div className="my-4">
                                  {material?.description || "---"}
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                    </div>
                  ))}
                {currentUser.isEducator && (
                  <div className="flex justify-center my-5">
                    <Link
                      to={`/class/${classId}/add-assignment-item?assignmentId=${assignment?._id}`}
                    >
                      <Button
                        variant="contained"
                        component="label"
                        style={{
                          backgroundColor: "#F8F8D5",
                          textTransform: "none",
                          color: "#000000",
                          border: "1px solid",
                        }}
                        size="large"
                      >
                        New Item <i className="fa-solid fa-plus fa-sm ml-2"></i>
                      </Button>
                    </Link>
                  </div>
                )}
              </Accordion>
            )
          ) : (
            <Accordion
              defaultExpanded
              className="mb-4"
              key={index}
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
                {isEditMode ? (
                  <div className="flex items-center justify-between w-full">
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
                    <div className="flex gap-2 items-center">
                      <Link
                        to={`/class/${classId}/edit-assignment/${assignment?._id}`}
                      >
                        <i className="fa-solid fa-pencil hover:text-red-600 text-sm"></i>
                      </Link>
                      {!assignment?.isHidden ? (
                        <i
                          className="fa-regular fa-eye hover:text-red-600 text-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowModalUpdateAssignmentIsHiddenTrue(true);
                            setAssignmentToUpdate(assignment?._id);
                          }}
                        ></i>
                      ) : (
                        <i
                          className="fa-regular fa-eye-slash hover:text-red-600 text-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowModalUpdateAssignmentIsHiddenFalse(true);
                            setAssignmentToUpdate(assignment?._id);
                          }}
                        ></i>
                      )}
                      <i
                        className="fa-solid fa-trash hover:text-red-600 text-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowModalDeleteAssignment(true);
                          setAssignmentToDelete(assignment?._id);
                        }}
                      ></i>
                    </div>
                  </div>
                ) : (
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
                )}
              </AccordionSummary>
              <div className="flex justify-between my-5 lg:w-11/12 mx-auto">
                <div className="flex flex-col gap-2">
                  <div className="flex gap-2">
                    <div className="flex flex-col gap-2">
                      <div className="font-bold">Open date:</div>
                      <div className="font-bold">Due date:</div>
                      <div className="font-bold">Description:</div>
                    </div>
                    <div className="flex flex-col gap-2">
                      <div className="font-bold text-red-600">
                        {assignment?.startDate
                          ? moment(assignment?.startDate).format(
                              "HH:mm:ss DD/MM/YYYY"
                            )
                          : "---"}
                      </div>
                      <div className="font-bold text-red-600">
                        {assignment?.endDate
                          ? moment(assignment?.endDate).format(
                              "HH:mm:ss DD/MM/YYYY"
                            )
                          : "---"}
                      </div>
                      <div className="">{assignment?.description || "---"}</div>
                    </div>
                  </div>
                </div>
                <Link
                  to={`/class/${classId}/view-submissions?assignmentId=${assignment?._id}`}
                >
                  <Button
                    variant="contained"
                    component="label"
                    style={{
                      backgroundColor: "#F8F8D5",
                      textTransform: "none",
                      color: "#000000",
                      border: "1px solid",
                    }}
                    size="large"
                  >
                    View Submissions
                  </Button>
                </Link>
              </div>
              {assignment?.materials?.length > 0 &&
                assignment?.materials.map((material, i) => (
                  <div
                    className="w-11/12 mx-auto border-2 border-gray-300 shadow-md p-4 mb-5 rounded-xl bg-gray-100 hover:bg-sky-100 transition"
                    key={i}
                  >
                    {material?.materialUrls?.length > 0 &&
                      material?.materialUrls.map((url, j, arr) => (
                        <div key={j} className="flex flex-col gap-4 pt-4">
                          <div className="flex items-center justify-between lg:w-11/12 mx-auto">
                            <div className="flex items-center gap-4">
                              <img
                                src={pdf}
                                alt="pdf icon"
                                className="w-6 h-6"
                              />
                              <Link
                                to={url}
                                target="_blank"
                                className="hover:underline text-cyan-600 ml-1"
                              >
                                {material?.nameFiles[j]?.substring(
                                  0,
                                  material?.nameFiles[j]?.length - 4
                                )}
                              </Link>
                            </div>
                            <div className="flex items-center gap-2">
                              {j === 0 && (
                                <div className="text-xs">
                                  Upload:{" "}
                                  {material?.createdAt
                                    ? moment(material?.createdAt).format(
                                        "DD/MM/YYYY"
                                      )
                                    : "---"}
                                </div>
                              )}
                              {isEditMode && j == 0 && (
                                <Link
                                  to={`/class/${classId}/edit-assignment-item/${material?._id}?assignmentId=${assignment?._id}`}
                                >
                                  <i className="fa-solid fa-pencil hover:text-red-600 text-sm"></i>
                                </Link>
                              )}
                              {isEditMode &&
                                j == 0 &&
                                (!material?.isHidden ? (
                                  <>
                                    <i
                                      className="fa-regular fa-eye cursor-pointer hover:text-red-600 text-sm"
                                      onClick={(e) => {
                                        setShowModalUpdateItemIsHiddenTrue(
                                          true
                                        );
                                        setItemToUpdate(material?._id);
                                      }}
                                    ></i>
                                    <i
                                      className="fa-solid fa-trash cursor-pointer hover:text-red-600 text-sm"
                                      onClick={() => {
                                        setShowModalDeleteItem(true);
                                        setItemToDelete(material?._id);
                                        setAssignmentIdOfItemToDelete(
                                          assignment?._id
                                        );
                                      }}
                                    ></i>
                                  </>
                                ) : (
                                  <>
                                    <i
                                      className="fa-regular fa-eye-slash cursor-pointer hover:text-red-600 text-sm"
                                      onClick={(e) => {
                                        setShowModalUpdateItemIsHiddenFalse(
                                          true
                                        );
                                        setItemToUpdate(material?._id);
                                      }}
                                    ></i>
                                    <i
                                      className="fa-solid fa-trash cursor-pointer hover:text-red-600 text-sm"
                                      onClick={() => {
                                        setShowModalDeleteItem(true);
                                        setItemToDelete(material?._id);
                                        setAssignmentIdOfItemToDelete(
                                          assignment?._id
                                        );
                                      }}
                                    ></i>
                                  </>
                                ))}
                            </div>
                          </div>
                          {j === arr.length - 1 && (
                            <div className="border-t border-black lg:w-11/12 mx-auto">
                              <div className="my-4">
                                {material?.description || "---"}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                  </div>
                ))}
              {currentUser.isEducator && (
                <div className="flex justify-center my-5">
                  <Link
                    to={`/class/${classId}/add-assignment-item?assignmentId=${assignment?._id}`}
                  >
                    <Button
                      variant="contained"
                      component="label"
                      style={{
                        backgroundColor: "#F8F8D5",
                        textTransform: "none",
                        color: "#000000",
                        border: "1px solid",
                      }}
                      size="large"
                    >
                      New Item <i className="fa-solid fa-plus fa-sm ml-2"></i>
                    </Button>
                  </Link>
                </div>
              )}
            </Accordion>
          )
        )}
      {currentUser.isEducator && (
        <div className="flex justify-center mb-5">
          <Link to={`/class/${classId}/add-assignment`}>
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
              New Assignment <i className="fa-solid fa-plus fa-sm ml-2"></i>
            </Button>
          </Link>
        </div>
      )}
      <Modal
        show={showModalUpdateAssignmentIsHiddenTrue}
        onClose={() => setShowModalUpdateAssignmentIsHiddenTrue(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to hide this assignment?
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
                onClick={handleUpdateAssignmentIsHiddenTrue}
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
                onClick={() => setShowModalUpdateAssignmentIsHiddenTrue(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalUpdateAssignmentIsHiddenFalse}
        onClose={() => setShowModalUpdateAssignmentIsHiddenFalse(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to unhide this assignment?
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
                onClick={handleUpdateAssignmentIsHiddenFalse}
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
                onClick={() => setShowModalUpdateAssignmentIsHiddenFalse(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalDeleteAssignment}
        onClose={() => setShowModalDeleteAssignment(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to delete this assignment?
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
                onClick={handleDeleteAssignment}
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
                onClick={() => setShowModalDeleteAssignment(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalUpdateItemIsHiddenTrue}
        onClose={() => setShowModalUpdateItemIsHiddenTrue(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to hide this item?
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
                onClick={handleUpdateItemIsHiddenTrue}
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
                onClick={() => setShowModalUpdateItemIsHiddenTrue(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalUpdateItemIsHiddenFalse}
        onClose={() => setShowModalUpdateItemIsHiddenFalse(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to unhide this item?
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
                onClick={handleUpdateItemIsHiddenFalse}
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
                onClick={() => setShowModalUpdateItemIsHiddenFalse(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalDeleteItem}
        onClose={() => setShowModalDeleteItem(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to delete this item?
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
                onClick={handleDeleteItem}
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
                onClick={() => setShowModalDeleteItem(false)}
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
