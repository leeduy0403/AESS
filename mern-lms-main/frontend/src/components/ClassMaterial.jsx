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
// import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
// import MenuBookIcon from "@mui/icons-material/MenuBook";
import {
  ExpandMore as ExpandMoreIcon,
  MenuBook as MenuBookIcon,
} from "@mui/icons-material";

export default function ClassMaterial() {
  const { classId } = useParams();
  const { currentUser } = useSelector((state) => state.user);
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const [sections, setSections] = useState([]);
  const [showModalDeleteSection, setShowModalDeleteSection] = useState(false);
  const [showModalDeleteItem, setShowModalDeleteItem] = useState(false);
  const [
    showModalUpdateSectionIsHiddenTrue,
    setShowModalUpdateSectionIsHiddenTrue,
  ] = useState(false);
  const [
    showModalUpdateSectionIsHiddenFalse,
    setShowModalUpdateSectionIsHiddenFalse,
  ] = useState(false);
  const [showModalUpdateItemIsHiddenTrue, setShowModalUpdateItemIsHiddenTrue] =
    useState(false);
  const [
    showModalUpdateItemIsHiddenFalse,
    setShowModalUpdateItemIsHiddenFalse,
  ] = useState(false);
  const [sectionToUpdate, setSectionToUpdate] = useState("");
  const [sectionToDelete, setSectionToDelete] = useState("");
  const [itemToUpdate, setItemToUpdate] = useState("");
  const [itemToDelete, setItemToDelete] = useState("");
  const [sectionIdOfItemToDelete, setSectionIdOfItemToDelete] = useState("");

  console.log(sections);

  useEffect(() => {
    const fetchSections = async () => {
      try {
        const res = await fetch(`/api/section/get/${classId}`);
        const data = await res.json();
        if (res.ok) {
          setSections(data);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchSections();
  }, [currentUser._id, classId]);

  const handleUpdateSectionIsHiddenTrue = async () => {
    setShowModalDeleteItem(false);
    try {
      const res = await fetch(
        `/api/section/update/${classId}/${sectionToUpdate}`,
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

  const handleUpdateSectionIsHiddenFalse = async () => {
    setShowModalDeleteItem(false);
    try {
      const res = await fetch(
        `/api/section/update/${classId}/${sectionToUpdate}`,
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
    setShowModalDeleteItem(false);
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
    setShowModalDeleteItem(false);
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

  const handleDeleteSection = async () => {
    setShowModalDeleteItem(false);
    try {
      const res = await fetch(
        `/api/section/delete/${classId}/${sectionToDelete}`,
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
        `/api/material/delete-material-section/${sectionIdOfItemToDelete}/${itemToDelete}`,
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
      {sections?.length > 0 &&
        sections.map((section, index) =>
          currentUser.isStudent ? (
            !section?.isHidden && (
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
                  <div className="flex gap-2">
                    <MenuBookIcon />
                    <Typography
                      component="span"
                      style={{ fontSize: "18px", fontWeight: "bold" }}
                    >
                      {section?.name}
                    </Typography>
                  </div>
                </AccordionSummary>
                <div className="flex gap-2 my-5 lg:w-11/12 mx-auto">
                  <div className="text-lg font-bold">Description: </div>
                  <div className="text-lg">{section?.description || "---"}</div>
                </div>
                {section?.materials?.length > 0 &&
                  section?.materials.map((material, i) => (
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
                                  underline="hover"
                                  target="_blank"
                                  className="hover:underline text-cyan-600"
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
                {currentUser?.isEducator && (
                  <div className="flex justify-center my-5">
                    <Link
                      to={`/class/${classId}/add-section-item?sectionId=${section?._id}`}
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
                      <MenuBookIcon />
                      <Typography
                        component="span"
                        style={{ fontSize: "18px", fontWeight: "bold" }}
                      >
                        {section?.name}
                      </Typography>
                    </div>
                    <div className="flex gap-2 items-center">
                      <Link
                        to={`/class/${classId}/edit-section/${section?._id}`}
                      >
                        <i className="fa-solid fa-pencil hover:text-red-600 text-sm"></i>
                      </Link>
                      {!section?.isHidden ? (
                        <i
                          className="fa-regular fa-eye hover:text-red-600 text-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowModalUpdateSectionIsHiddenTrue(true);
                            setSectionToUpdate(section?._id);
                          }}
                        ></i>
                      ) : (
                        <i
                          className="fa-regular fa-eye-slash hover:text-red-600 text-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowModalUpdateSectionIsHiddenFalse(true);
                            setSectionToUpdate(section?._id);
                          }}
                        ></i>
                      )}
                      <i
                        className="fa-solid fa-trash hover:text-red-600 text-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowModalDeleteSection(true);
                          setSectionToDelete(section?._id);
                        }}
                      ></i>
                    </div>
                  </div>
                ) : (
                  <div className="flex gap-2">
                    <MenuBookIcon />
                    <Typography
                      component="span"
                      style={{ fontSize: "18px", fontWeight: "bold" }}
                    >
                      {section?.name}
                    </Typography>
                  </div>
                )}
              </AccordionSummary>
              <div className="flex gap-2 my-5 lg:w-11/12 mx-auto">
                <div className="text-lg font-bold">Description: </div>
                <div className="text-lg">{section?.description || "---"}</div>
              </div>
              {section?.materials?.length > 0 &&
                section?.materials.map((material, i) => (
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
                                underline="hover"
                                target="_blank"
                                className="hover:underline text-cyan-600"
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
                                  to={`/class/${classId}/edit-section-item/${material?._id}?sectionId=${section?._id}`}
                                >
                                  <i className="fa-solid fa-pencil hover:text-red-600 text-sm"></i>
                                </Link>
                              )}
                              {isEditMode &&
                                j == 0 &&
                                (!material?.isHidden ? (
                                  <i
                                    className="fa-regular fa-eye cursor-pointer hover:text-red-600 text-sm"
                                    onClick={(e) => {
                                      setShowModalUpdateItemIsHiddenTrue(true);
                                      setItemToUpdate(material?._id);
                                    }}
                                  ></i>
                                ) : (
                                  <i
                                    className="fa-regular fa-eye-slash cursor-pointer hover:text-red-600 text-sm"
                                    onClick={(e) => {
                                      setShowModalUpdateItemIsHiddenFalse(true);
                                      setItemToUpdate(material?._id);
                                    }}
                                  ></i>
                                ))}
                              {isEditMode && j == 0 && (
                                <i
                                  className="fa-solid fa-trash cursor-pointer hover:text-red-600 text-sm"
                                  onClick={() => {
                                    setShowModalDeleteItem(true);
                                    setItemToDelete(material?._id);
                                    setSectionIdOfItemToDelete(section?._id);
                                  }}
                                ></i>
                              )}
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
              {currentUser?.isEducator && (
                <div className="flex justify-center my-5">
                  <Link
                    to={`/class/${classId}/add-section-item?sectionId=${section?._id}`}
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
                    >
                      New Item <i className="fa-solid fa-plus fa-sm ml-2"></i>
                    </Button>
                  </Link>
                </div>
              )}
            </Accordion>
          )
        )}
      {currentUser?.isEducator && (
        <div className="flex justify-center mb-5">
          <Link to={`/class/${classId}/add-section`}>
            <Button
              variant="contained"
              component="label"
              style={{
                backgroundColor: "#26597C",
                textTransform: "none",
              }}
            >
              New Section <i className="fa-solid fa-plus fa-sm ml-2"></i>
            </Button>
          </Link>
        </div>
      )}
      <Modal
        show={showModalUpdateSectionIsHiddenTrue}
        onClose={() => setShowModalUpdateSectionIsHiddenTrue(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to hide this section?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  textTransform: "none",
                }}
                color="error"
                onClick={handleUpdateSectionIsHiddenTrue}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  textTransform: "none",
                }}
                onClick={() => setShowModalUpdateSectionIsHiddenTrue(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalUpdateSectionIsHiddenFalse}
        onClose={() => setShowModalUpdateSectionIsHiddenFalse(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to unhide this section?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  textTransform: "none",
                }}
                color="error"
                onClick={handleUpdateSectionIsHiddenFalse}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  textTransform: "none",
                }}
                onClick={() => setShowModalUpdateSectionIsHiddenFalse(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showModalDeleteSection}
        onClose={() => setShowModalDeleteSection(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to delete this section?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  textTransform: "none",
                }}
                color="error"
                onClick={handleDeleteSection}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  textTransform: "none",
                }}
                onClick={() => setShowModalDeleteSection(false)}
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
                  textTransform: "none",
                }}
                color="error"
                onClick={handleUpdateItemIsHiddenTrue}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
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
                  textTransform: "none",
                }}
                color="error"
                onClick={handleUpdateItemIsHiddenFalse}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
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
                  textTransform: "none",
                }}
                color="error"
                onClick={handleDeleteItem}
                fullWidth
              >
                Yes, I'm sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
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
